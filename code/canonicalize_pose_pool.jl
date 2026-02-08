#!/usr/bin/env julia

using Printf

include("filter_identical_poses.jl")

struct CanonCli
    input::String
    output::String
    max_poses_per_chunk::Int
    keep_temp::Bool
end

function canon_usage()
    println(
        """
        Usage:
          canonicalize_pose_pool.jl <input_dir> <output_dir> [options]

        Options:
          --max-poses-per-chunk <int>  Maximum poses per output chunk (default: 4294967295)
          --keep-temp                  Keep <output_dir>/_tmp
          -h, --help                   Show help
        """
    )
end

function parse_canon_args(argv::Vector{String})::CanonCli
    if any(a -> a == "-h" || a == "--help", argv)
        canon_usage()
        exit(0)
    end
    length(argv) >= 2 || error("Expected <input_dir> and <output_dir>")
    input = argv[1]
    output = argv[2]
    max_poses_per_chunk = 5_000_000
    keep_temp = false

    i = 3
    while i <= length(argv)
        arg = argv[i]
        if arg == "--max-poses-per-chunk"
            i + 1 <= length(argv) || error("--max-poses-per-chunk requires a value")
            max_poses_per_chunk = parse(Int, argv[i + 1])
            i += 2
        elseif arg == "--keep-temp"
            keep_temp = true
            i += 1
        else
            error("Unknown argument: $arg")
        end
    end
    max_poses_per_chunk > 0 || error("--max-poses-per-chunk must be positive")
    return CanonCli(input, output, max_poses_per_chunk, keep_temp)
end

function write_offsets_file_fixed(
    path::String,
    center::NTuple{3, Int16},
    offsets::Vector{NTuple{3, Int8}},
)
    open(path, "w") do io
        write_le_u16(io, i16_to_u16(center[1]))
        write_le_u16(io, i16_to_u16(center[2]))
        write_le_u16(io, i16_to_u16(center[3]))
        @inbounds for (dx, dy, dz) in offsets
            write(io, i8_to_u8(dx))
            write(io, i8_to_u8(dy))
            write(io, i8_to_u8(dz))
        end
    end
end

function for_each_key_in_file(f::Function, path::String)
    if !isfile(path)
        return
    end
    open_io = endswith(path, ".zst") ? (`zstd -q -dc -- $path`) : path
    open(open_io, "r") do io
        buf = Vector{UInt8}(undef, 1 << 20)
        rem = UInt8[]
        while true
            n = readbytes!(io, buf)
            n == 0 && break
            data = if isempty(rem)
                Vector{UInt8}(view(buf, 1:n))
            else
                tmp = Vector{UInt8}(undef, length(rem) + n)
                copyto!(tmp, 1, rem, 1, length(rem))
                copyto!(tmp, length(rem) + 1, buf, 1, n)
                tmp
            end
            full = (length(data) ÷ 8) * 8
            p = 1
            @inbounds while p <= full
                f(read_le_u64(data, p))
                p += 8
            end
            if full < length(data)
                rem = data[(full + 1):end]
            else
                empty!(rem)
            end
        end
        isempty(rem) || error("Corrupt key stream size (not multiple of 8): $path")
    end
end

function flush_canon_chunk!(
    outdir::String,
    next_index::Base.RefValue{Int},
    center::NTuple{3, Int16},
    conf::Vector{UInt16},
    rot::Vector{UInt16},
    offidx::Vector{UInt16},
    offsets::Vector{NTuple{3, Int8}},
)
    isempty(conf) && return
    idx = next_index[]
    pose_path = joinpath(outdir, "poses-$(idx).npy.zst")
    offs_path = joinpath(outdir, "offsets-$(idx).dat")
    write_npy_u16_3cols_zst(pose_path, conf, rot, offidx)
    write_offsets_file_fixed(offs_path, center, offsets)
    next_index[] = idx + 1
    empty!(conf)
    empty!(rot)
    empty!(offidx)
    empty!(offsets)
end

function canonicalize_index_to_output!(
    index::CanonicalSetIndex,
    outdir::String,
    max_poses_per_chunk::Int,
)
    mkpath(outdir)
    centers = sort!(collect(collect_center_keys(index)))
    next_index = Ref(1)
    max_offsets = Int(typemax(UInt16)) + 1

    for (ci, center) in enumerate(centers)
        conf = UInt16[]
        rot = UInt16[]
        offidx = UInt16[]
        offsets = NTuple{3, Int8}[]
        offset_map = Dict{NTuple{3, Int8}, UInt16}()

        function add_pose_rel!(c::UInt16, r::UInt16, rel::NTuple{3, Int8})
            if length(conf) >= max_poses_per_chunk || (!haskey(offset_map, rel) && length(offsets) >= max_offsets)
                flush_canon_chunk!(outdir, next_index, center, conf, rot, offidx, offsets)
                empty!(offset_map)
            end
            oi = get(offset_map, rel, UInt16(0xffff))
            if oi == UInt16(0xffff)
                push!(offsets, rel)
                oi = UInt16(length(offsets) - 1)
                offset_map[rel] = oi
            end
            push!(conf, c)
            push!(rot, r)
            push!(offidx, oi)
        end

        pair_refs = get(index.pairs_by_center, center, PairWithStart[])
        for pref in pair_refs
            lp = load_pair(pref.pair)
            lp.center == center || error("Center mismatch for $(pref.pair.pose_path)")
            n = length(lp.conf)
            @inbounds for i in 1:n
                oi = Int(lp.offidx[i]) + 1
                rel = (lp.rel_offsets[oi, 1], lp.rel_offsets[oi, 2], lp.rel_offsets[oi, 3])
                add_pose_rel!(lp.conf[i], lp.rot[i], rel)
            end
        end

        for path in get(index.bucket_by_center, center, String[])
            for_each_key_in_file(path) do key
                c, r, rx, ry, rz = unpack_rel_pose_key(key)
                add_pose_rel!(c, r, (rx, ry, rz))
            end
        end

        flush_canon_chunk!(outdir, next_index, center, conf, rot, offidx, offsets)
        println(@sprintf("center %d/%d done: (%d,%d,%d)", ci, length(centers), center[1], center[2], center[3]))
    end

    if next_index[] == 1
        # Keep output shape consistent with other tools.
        write_npy_u16_3cols_zst(joinpath(outdir, "poses-1.npy.zst"), UInt16[], UInt16[], UInt16[])
        write_offsets_file_fixed(joinpath(outdir, "offsets-1.dat"), (Int16(0), Int16(0), Int16(0)), NTuple{3, Int8}[])
    end
end

function main(argv::Vector{String})::Int
    cfg = parse_canon_args(argv)
    isdir(cfg.input) || error("Input directory not found: $(cfg.input)")
    !ispath(cfg.output) || error("Output path already exists: $(cfg.output)")

    mkpath(cfg.output)
    tmp_root = joinpath(cfg.output, "_tmp")
    mkpath(tmp_root)

    pairs = discover_pose_pairs(cfg.input)
    println("input pairs: $(length(pairs))")
    println("stage:start canonicalization-index")
    t0 = time()
    index = build_canonical_index!(pairs, tmp_root, nothing)
    println(@sprintf("stage:end canonicalization-index %.3f", time() - t0))
    println("indexed poses: $(index.total_poses), temp rewrite (raw est): $(fmt_bytes(index.temp_bytes))")

    println("stage:start write-canonical-output")
    t1 = time()
    canonicalize_index_to_output!(index, cfg.output, cfg.max_poses_per_chunk)
    println(@sprintf("stage:end write-canonical-output %.3f", time() - t1))
    println("done")

    if !cfg.keep_temp
        rm(tmp_root; recursive = true, force = true)
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(copy(ARGS)))
end
