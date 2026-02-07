#!/usr/bin/env julia

using Base.Threads
using Printf

const I16_MIN = Int32(typemin(Int8))
const I16_MAX = Int32(typemax(Int8))

struct CliConfig
    set1::String
    set2::String
    output::String
    threshold_k::Int
    threshold_m::Int
    max_poses_per_chunk::Int
    dedup_buckets::Int
    keep_temp::Bool
end

struct PairInfo
    index::Int
    pose_path::String
    offset_path::String
    pose_file_size::Int64
end

struct LoadedPair
    conf::Vector{UInt16}
    rot::Vector{UInt16}
    offidx::Vector{UInt16}
    center::NTuple{3, Int16}
    rel_offsets::Matrix{Int8}
    abs_offsets::Matrix{Int16}
end

mutable struct ActivePool
    conf::Vector{UInt16}
    rot::Vector{UInt16}
    x::Vector{Int16}
    y::Vector{Int16}
    z::Vector{Int16}
    aidx::Vector{UInt32}
end

struct MatchResultFirstPass
    conf::Vector{UInt16}
    rot::Vector{UInt16}
    x::Vector{Int16}
    y::Vector{Int16}
    z::Vector{Int16}
    aidx::Vector{UInt32}
end

struct MatchResultSecondPass
    aidx::Vector{UInt32}
    uniq_idx::Vector{UInt32}
end

struct BLookupFirstPass
    center::NTuple{3, Int16}
    rel_to_indices::Dict{UInt32, Vector{UInt16}}
    key_counts::Dict{UInt64, Int32}
end

struct BLookupSecondPass
    center::NTuple{3, Int16}
    rel_to_indices::Dict{UInt32, Vector{UInt16}}
    key_to_unique::Dict{UInt64, UInt32}
end

struct BucketRecord
    conf::UInt16
    rot::UInt16
    x::Int16
    y::Int16
    z::Int16
    old_idx::UInt32
end

struct MapRecord
    old_idx::UInt32
    uniq_idx::UInt32
end

struct KeyRecord
    conf::UInt16
    rot::UInt16
    x::Int16
    y::Int16
    z::Int16
end

struct HeapNode
    old_idx::UInt32
    uniq_idx::UInt32
    src::Int
end

mutable struct PoseFileWriter
    outdir::String
    next_index::Int
    max_poses_per_chunk::Int
    write_empty_if_none::Bool
    total_written::UInt64
    conf::Vector{UInt16}
    rot::Vector{UInt16}
    offidx::Vector{UInt16}
    offset_map::Dict{NTuple{3, Int16}, UInt16}
    offsets::Vector{NTuple{3, Int16}}
    min_xyz::Union{Nothing, NTuple{3, Int32}}
    max_xyz::Union{Nothing, NTuple{3, Int32}}
end

mutable struct IndexFileWriter
    outdir::String
    prefix::String
    next_index::Int
    ncols::Int
    flush_rows::Int
    total_rows::UInt64
    col1::Vector{UInt32}
    col2::Vector{UInt32}
end

struct PendingRawK
    conf::Vector{UInt16}
    rot::Vector{UInt16}
    x::Vector{Int16}
    y::Vector{Int16}
    z::Vector{Int16}
    aidx::Vector{UInt32}
end

struct FirstPassResult
    k_count::UInt64
    pending::Union{Nothing, PendingRawK}
end

function usage()
    println(
        """
        Usage:
          filter_identical_poses.jl <set1> <set2> <output> [options]

        Options:
          --threshold-k <int>         Flush threshold for matched rows (default: 2000000)
          --threshold-m <int>         Reserve threshold for active pool (default: 200000)
          --max-poses-per-chunk <int> Maximum poses per output poses-*.npy chunk (default: 4294967295)
          --dedup-buckets <int>       Power-of-two bucket count for K uniqueness (default: 256)
          --keep-temp                 Keep temporary files under <output>/_tmp for inspection
          -h, --help                  Show help
        """
    )
end

function parse_args(argv::Vector{String})::CliConfig
    if any(a -> a == "-h" || a == "--help", argv)
        usage()
        exit(0)
    end
    if length(argv) < 3
        usage()
        error("Expected at least 3 positional arguments")
    end

    set1 = argv[1]
    set2 = argv[2]
    output = argv[3]

    threshold_k = 2_000_000
    threshold_m = 200_000
    max_poses_per_chunk = Int(typemax(UInt32))
    dedup_buckets = 256
    keep_temp = false

    i = 4
    while i <= length(argv)
        arg = argv[i]
        if arg == "--threshold-k"
            i + 1 <= length(argv) || error("--threshold-k requires a value")
            threshold_k = parse(Int, argv[i + 1])
            i += 2
            continue
        elseif arg == "--threshold-m"
            i + 1 <= length(argv) || error("--threshold-m requires a value")
            threshold_m = parse(Int, argv[i + 1])
            i += 2
            continue
        elseif arg == "--max-poses-per-chunk"
            i + 1 <= length(argv) || error("--max-poses-per-chunk requires a value")
            max_poses_per_chunk = parse(Int, argv[i + 1])
            i += 2
            continue
        elseif arg == "--dedup-buckets"
            i + 1 <= length(argv) || error("--dedup-buckets requires a value")
            dedup_buckets = parse(Int, argv[i + 1])
            i += 2
            continue
        elseif arg == "--keep-temp"
            keep_temp = true
            i += 1
            continue
        else
            error("Unknown argument: $arg")
        end
    end

    threshold_k > 0 || error("--threshold-k must be positive")
    threshold_m > 0 || error("--threshold-m must be positive")
    max_poses_per_chunk > 0 || error("--max-poses-per-chunk must be positive")
    dedup_buckets > 0 || error("--dedup-buckets must be positive")
    (dedup_buckets & (dedup_buckets - 1)) == 0 || error("--dedup-buckets must be a power of two")

    return CliConfig(
        set1,
        set2,
        output,
        threshold_k,
        threshold_m,
        max_poses_per_chunk,
        dedup_buckets,
        keep_temp,
    )
end

function read_le_u16(bytes::Vector{UInt8}, pos::Int)::UInt16
    return UInt16(bytes[pos]) | (UInt16(bytes[pos + 1]) << 8)
end

function read_le_u32(bytes::Vector{UInt8}, pos::Int)::UInt32
    return (
        UInt32(bytes[pos]) |
        (UInt32(bytes[pos + 1]) << 8) |
        (UInt32(bytes[pos + 2]) << 16) |
        (UInt32(bytes[pos + 3]) << 24)
    )
end

function write_le_u16(io::IO, x::UInt16)
    write(io, UInt8(x & 0x00ff))
    write(io, UInt8((x >> 8) & 0x00ff))
end

function write_le_u32(io::IO, x::UInt32)
    write(io, UInt8(x & 0x000000ff))
    write(io, UInt8((x >> 8) & 0x000000ff))
    write(io, UInt8((x >> 16) & 0x000000ff))
    write(io, UInt8((x >> 24) & 0x000000ff))
end

function u8_to_i8(x::UInt8)::Int8
    return x <= 0x7f ? Int8(x) : Int8(Int16(x) - 256)
end

function i8_to_u8(x::Int8)::UInt8
    return x >= 0 ? UInt8(x) : UInt8(Int16(x) + 256)
end

function u16_to_i16(x::UInt16)::Int16
    return reinterpret(Int16, x)
end

function i16_to_u16(x::Int16)::UInt16
    return reinterpret(UInt16, x)
end

@inline function pack_rel_key(dx::Int8, dy::Int8, dz::Int8)::UInt32
    return (
        UInt32(reinterpret(UInt8, dx)) |
        (UInt32(reinterpret(UInt8, dy)) << 8) |
        (UInt32(reinterpret(UInt8, dz)) << 16)
    )
end

@inline function pack_pose_key(conf::UInt16, rot::UInt16, offidx::UInt16)::UInt64
    return UInt64(conf) | (UInt64(rot) << 16) | (UInt64(offidx) << 32)
end

function dtype_descr(::Type{UInt16})::String
    return "<u2"
end

function dtype_descr(::Type{UInt32})::String
    return "<u4"
end

function dtype_descr(::Type{Int16})::String
    return "<i2"
end

function parse_shape_tuple(shape_text::String)::Vector{Int}
    stripped = strip(shape_text)
    isempty(stripped) && return Int[]
    parts = split(stripped, ",")
    dims = Int[]
    for p in parts
        q = strip(p)
        isempty(q) && continue
        push!(dims, parse(Int, q))
    end
    return dims
end

function parse_npy_dict_header(header::String)
    m_descr = match(r"'descr':\s*'([^']+)'", header)
    m_fortran = match(r"'fortran_order':\s*(True|False)", header)
    m_shape = match(r"'shape':\s*\(([^)]*)\)", header)
    m_descr === nothing && error("Unable to parse .npy descr")
    m_fortran === nothing && error("Unable to parse .npy fortran_order")
    m_shape === nothing && error("Unable to parse .npy shape")
    descr = m_descr.captures[1]
    fortran = m_fortran.captures[1] == "True"
    shape = parse_shape_tuple(String(m_shape.captures[1]))
    return descr, fortran, shape
end

function parse_npy_header(bytes::Vector{UInt8})
    length(bytes) >= 10 || error("Invalid .npy file (too small)")
    magic = bytes[1:6]
    magic == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
        error("Invalid .npy magic")

    major = Int(bytes[7])
    minor = Int(bytes[8])
    header_len = 0
    start = 0
    if major == 1
        header_len = Int(read_le_u16(bytes, 9))
        start = 11
    elseif major == 2
        header_len = Int(read_le_u32(bytes, 9))
        start = 13
    else
        error("Unsupported .npy version: $major.$minor")
    end
    stop = start + header_len - 1
    stop <= length(bytes) || error("Invalid .npy header length")
    header = String(bytes[start:stop])
    descr, fortran, shape = parse_npy_dict_header(header)
    return descr, fortran, shape, stop + 1
end

function parse_npy_header_io(io::IO)
    magic = read(io, 6)
    magic == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
        error("Invalid .npy magic")
    major = Int(read(io, UInt8))
    _minor = Int(read(io, UInt8))

    header_len = if major == 1
        b = read(io, 2)
        Int(read_le_u16(b, 1))
    elseif major == 2
        b = read(io, 4)
        Int(read_le_u32(b, 1))
    else
        error("Unsupported .npy version: $major")
    end
    header = String(read(io, header_len))
    return parse_npy_dict_header(header)
end

function read_npy_u16(path::String)
    bytes = read(path)
    return read_npy_u16_bytes(bytes)
end

function read_npy_u16_bytes(bytes::Vector{UInt8})
    descr, fortran, shape, data_pos = parse_npy_header(bytes)
    (descr == "<u2" || descr == "|u2") || error("Expected uint16 .npy, got $descr")
    fortran && error("Fortran-order arrays are not supported")
    (length(shape) == 1 || length(shape) == 2) || error("Unsupported rank $(length(shape))")
    nelem = prod(shape)
    needed = nelem * 2
    data_end = data_pos + needed - 1
    data_end <= length(bytes) || error("Unexpected EOF in .npy payload")

    raw = Vector{UInt16}(undef, nelem)
    p = data_pos
    @inbounds for i in 1:nelem
        raw[i] = read_le_u16(bytes, p)
        p += 2
    end

    if length(shape) == 1
        return raw
    else
        nrows, ncols = shape
        mat = Matrix{UInt16}(undef, nrows, ncols)
        idx = 1
        @inbounds for i in 1:nrows
            for j in 1:ncols
                mat[i, j] = raw[idx]
                idx += 1
            end
        end
        return mat
    end
end

function read_npy_u32(path::String)
    bytes = read(path)
    descr, fortran, shape, data_pos = parse_npy_header(bytes)
    (descr == "<u4" || descr == "|u4") || error("Expected uint32 .npy, got $descr")
    fortran && error("Fortran-order arrays are not supported")
    (length(shape) == 1 || length(shape) == 2) || error("Unsupported rank $(length(shape))")
    nelem = prod(shape)
    needed = nelem * 4
    data_end = data_pos + needed - 1
    data_end <= length(bytes) || error("Unexpected EOF in .npy payload")

    raw = Vector{UInt32}(undef, nelem)
    p = data_pos
    @inbounds for i in 1:nelem
        raw[i] = read_le_u32(bytes, p)
        p += 4
    end

    if length(shape) == 1
        return raw
    else
        nrows, ncols = shape
        mat = Matrix{UInt32}(undef, nrows, ncols)
        idx = 1
        @inbounds for i in 1:nrows
            for j in 1:ncols
                mat[i, j] = raw[idx]
                idx += 1
            end
        end
        return mat
    end
end

function write_npy_header(io::IO, descr::String, shape::Vector{Int})
    shape_text = if length(shape) == 1
        "(" * string(shape[1]) * ",)"
    else
        "(" * join(shape, ", ") * ",)"
    end
    base = "{'descr': '$descr', 'fortran_order': False, 'shape': $shape_text, }"
    preamble_len = 10
    header = base
    while (preamble_len + length(header) + 1) % 16 != 0
        header *= " "
    end
    header *= "\n"
    header_bytes = Vector{UInt8}(codeunits(header))
    length(header_bytes) <= 0xffff || error("Header too large for .npy v1.0")

    write(io, UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')])
    write(io, UInt8(1))
    write(io, UInt8(0))
    write_le_u16(io, UInt16(length(header_bytes)))
    write(io, header_bytes)
end

function write_npy_u16(path::String, arr::AbstractArray{UInt16})
    open(path, "w") do io
        if ndims(arr) == 1
            write_npy_header(io, dtype_descr(UInt16), [length(arr)])
            @inbounds for v in arr
                write_le_u16(io, v)
            end
        elseif ndims(arr) == 2
            nrows, ncols = size(arr)
            write_npy_header(io, dtype_descr(UInt16), [nrows, ncols])
            @inbounds for i in 1:nrows
                for j in 1:ncols
                    write_le_u16(io, arr[i, j])
                end
            end
        else
            error("write_npy_u16 supports only rank-1 or rank-2 arrays")
        end
    end
end

function write_npy_u32(path::String, arr::AbstractArray{UInt32})
    open(path, "w") do io
        if ndims(arr) == 1
            write_npy_header(io, dtype_descr(UInt32), [length(arr)])
            @inbounds for v in arr
                write_le_u32(io, v)
            end
        elseif ndims(arr) == 2
            nrows, ncols = size(arr)
            write_npy_header(io, dtype_descr(UInt32), [nrows, ncols])
            @inbounds for i in 1:nrows
                for j in 1:ncols
                    write_le_u32(io, arr[i, j])
                end
            end
        else
            error("write_npy_u32 supports only rank-1 or rank-2 arrays")
        end
    end
end

function write_npy_u16_3cols(
    path::String,
    col1::Vector{UInt16},
    col2::Vector{UInt16},
    col3::Vector{UInt16},
)
    n = length(col1)
    length(col2) == n || error("Column length mismatch")
    length(col3) == n || error("Column length mismatch")
    raw = Vector{UInt16}(undef, n * 3)
    @inbounds for i in 1:n
        j = 3 * (i - 1)
        raw[j + 1] = col1[i]
        raw[j + 2] = col2[i]
        raw[j + 3] = col3[i]
    end
    open(path, "w") do io
        write_npy_header(io, dtype_descr(UInt16), [n, 3])
        @inbounds for v in raw
            write_le_u16(io, v)
        end
    end
end

function write_npy_u32_1col(path::String, col1::Vector{UInt32})
    n = length(col1)
    open(path, "w") do io
        write_npy_header(io, dtype_descr(UInt32), [n, 1])
        @inbounds for v in col1
            write_le_u32(io, v)
        end
    end
end

function write_npy_u32_2cols(path::String, col1::Vector{UInt32}, col2::Vector{UInt32})
    n = length(col1)
    length(col2) == n || error("Column length mismatch")
    raw = Vector{UInt32}(undef, n * 2)
    @inbounds for i in 1:n
        j = 2 * (i - 1)
        raw[j + 1] = col1[i]
        raw[j + 2] = col2[i]
    end
    open(path, "w") do io
        write_npy_header(io, dtype_descr(UInt32), [n, 2])
        @inbounds for v in raw
            write_le_u32(io, v)
        end
    end
end

function decompress_npy_zst(path::String)::Vector{UInt8}
    cmd = `zstd -dc -- $path`
    try
        return read(cmd)
    catch err
        error("Failed to read $path with zstd -dc: $err")
    end
end

function read_pose_columns_bytes(bytes::Vector{UInt8}, src::String)
    descr, fortran, shape, data_pos = parse_npy_header(bytes)
    (descr == "<u2" || descr == "|u2") || error("Expected uint16 .npy in $src, got $descr")
    fortran && error("Fortran-order arrays are not supported in $src")
    length(shape) == 2 || error("Expected 2D pose array in $src, got rank $(length(shape))")
    shape[2] == 3 || error("Expected pose shape [N,3] in $src, got $(shape)")

    nrows = shape[1]
    needed = nrows * 3 * 2
    data_end = data_pos + needed - 1
    data_end <= length(bytes) || error("Unexpected EOF in .npy payload: $src")

    conf = Vector{UInt16}(undef, nrows)
    rot = Vector{UInt16}(undef, nrows)
    offidx = Vector{UInt16}(undef, nrows)
    p = data_pos
    @inbounds for i in 1:nrows
        conf[i] = read_le_u16(bytes, p)
        rot[i] = read_le_u16(bytes, p + 2)
        offidx[i] = read_le_u16(bytes, p + 4)
        p += 6
    end
    return conf, rot, offidx
end

function read_pose_columns(path::String)
    if endswith(path, ".npy.zst")
        bytes = decompress_npy_zst(path)
        return read_pose_columns_bytes(bytes, path)
    end
    bytes = read(path)
    return read_pose_columns_bytes(bytes, path)
end

function pose_row_count(path::String)::Int
    if endswith(path, ".npy")
        open(path, "r") do io
            descr, fortran, shape = parse_npy_header_io(io)
            (descr == "<u2" || descr == "|u2") || error("Expected uint16 .npy in $path, got $descr")
            fortran && error("Fortran-order arrays are not supported in $path")
            length(shape) == 2 || error("Expected 2D pose array in $path")
            shape[2] == 3 || error("Expected pose shape [N,3] in $path")
            return shape[1]
        end
    end
    conf, _, _ = read_pose_columns(path)
    return length(conf)
end

function read_pose_npy(path::String)
    if endswith(path, ".npy.zst")
        bytes = decompress_npy_zst(path)
        return read_npy_u16_bytes(bytes)
    end
    return read_npy_u16(path)
end

function discover_pose_pairs(dir::String)::Vector{PairInfo}
    isdir(dir) || error("Pose directory not found: $dir")
    entries = readdir(dir)
    indexed = Dict{Int, String}()
    for name in entries
        if !startswith(name, "poses-")
            continue
        end
        idx::Union{Nothing, Int} = nothing
        if endswith(name, ".npy")
            text = name[7:(end - 4)]
            if all(isdigit, text)
                idx = parse(Int, text)
            end
        elseif endswith(name, ".npy.zst")
            text = name[7:(end - 8)]
            if all(isdigit, text)
                idx = parse(Int, text)
            end
        end
        idx === nothing && continue
        old = get(indexed, idx, "")
        if isempty(old)
            indexed[idx] = name
        elseif endswith(old, ".zst") && endswith(name, ".npy")
            indexed[idx] = name
        end
    end

    isempty(indexed) && error("No pose files found in $dir")

    out = PairInfo[]
    for idx in sort(collect(keys(indexed)))
        pose_name = indexed[idx]
        pose_path = joinpath(dir, pose_name)
        offset_path = joinpath(dir, "offsets-$idx.dat")
        isfile(offset_path) || error("Missing $offset_path for $pose_path")
        sz = stat(pose_path).size
        push!(out, PairInfo(idx, pose_path, offset_path, sz))
    end
    return out
end

function load_offsets_dat(path::String)
    bytes = read(path)
    length(bytes) >= 6 || error("offset file too small: $path")
    c1 = u16_to_i16(read_le_u16(bytes, 1))
    c2 = u16_to_i16(read_le_u16(bytes, 3))
    c3 = u16_to_i16(read_le_u16(bytes, 5))
    rem = length(bytes) - 6
    rem % 3 == 0 || error("offset file payload length must be divisible by 3: $path")
    n = rem ÷ 3
    rel = Matrix{Int8}(undef, n, 3)
    abs = Matrix{Int16}(undef, n, 3)
    p = 7
    @inbounds for i in 1:n
        dx = u8_to_i8(bytes[p])
        dy = u8_to_i8(bytes[p + 1])
        dz = u8_to_i8(bytes[p + 2])
        rel[i, 1] = dx
        rel[i, 2] = dy
        rel[i, 3] = dz
        abs[i, 1] = Int16(Int32(c1) + Int32(dx))
        abs[i, 2] = Int16(Int32(c2) + Int32(dy))
        abs[i, 3] = Int16(Int32(c3) + Int32(dz))
        p += 3
    end
    return (Int16(c1), Int16(c2), Int16(c3)), rel, abs
end

function load_pair(pair::PairInfo)::LoadedPair
    conf, rot, offidx = read_pose_columns(pair.pose_path)
    center, rel_offsets, abs_offsets = load_offsets_dat(pair.offset_path)
    n = length(conf)
    max_off = size(abs_offsets, 1)
    @inbounds for i in 1:n
        Int(offidx[i]) + 1 <= max_off || error("Offset index out of range in $(pair.pose_path)")
    end
    return LoadedPair(conf, rot, offidx, center, rel_offsets, abs_offsets)
end

function init_active_pool()::ActivePool
    return ActivePool(UInt16[], UInt16[], Int16[], Int16[], Int16[], UInt32[])
end

function active_length(a::ActivePool)::Int
    return length(a.conf)
end

function append_active!(dst::ActivePool, src::ActivePool)
    append!(dst.conf, src.conf)
    append!(dst.rot, src.rot)
    append!(dst.x, src.x)
    append!(dst.y, src.y)
    append!(dst.z, src.z)
    append!(dst.aidx, src.aidx)
end

function clear_active!(a::ActivePool)
    empty!(a.conf)
    empty!(a.rot)
    empty!(a.x)
    empty!(a.y)
    empty!(a.z)
    empty!(a.aidx)
end

function add_new_aa_to_active!(
    active::ActivePool,
    pair::LoadedPair,
    a_counter::UInt64,
)
    n = length(pair.conf)
    n <= typemax(UInt32) || error("Single AA pair exceeds uint32 indexing range")
    a_counter + UInt64(n) <= UInt64(typemax(UInt32)) + UInt64(1) ||
        error("A global index exceeds uint32 range")
    @inbounds for i in 1:n
        oi = Int(pair.offidx[i]) + 1
        push!(active.conf, pair.conf[i])
        push!(active.rot, pair.rot[i])
        push!(active.x, pair.abs_offsets[oi, 1])
        push!(active.y, pair.abs_offsets[oi, 2])
        push!(active.z, pair.abs_offsets[oi, 3])
        push!(active.aidx, UInt32(a_counter + UInt64(i - 1)))
    end
end

function make_blookup_first(pair::LoadedPair)::BLookupFirstPass
    rel_to_indices = Dict{UInt32, Vector{UInt16}}()
    noff = size(pair.rel_offsets, 1)
    @inbounds for i in 1:noff
        key = pack_rel_key(pair.rel_offsets[i, 1], pair.rel_offsets[i, 2], pair.rel_offsets[i, 3])
        vec = get!(rel_to_indices, key, UInt16[])
        push!(vec, UInt16(i - 1))
    end

    n = length(pair.conf)
    key_counts = Dict{UInt64, Int32}()
    sizehint!(key_counts, n)
    @inbounds for i in 1:n
        k = pack_pose_key(pair.conf[i], pair.rot[i], pair.offidx[i])
        key_counts[k] = get(key_counts, k, Int32(0)) + Int32(1)
    end
    return BLookupFirstPass(pair.center, rel_to_indices, key_counts)
end

function make_blookup_second(
    pair::LoadedPair,
    global_start::UInt32,
)::BLookupSecondPass
    rel_to_indices = Dict{UInt32, Vector{UInt16}}()
    noff = size(pair.rel_offsets, 1)
    @inbounds for i in 1:noff
        key = pack_rel_key(pair.rel_offsets[i, 1], pair.rel_offsets[i, 2], pair.rel_offsets[i, 3])
        vec = get!(rel_to_indices, key, UInt16[])
        push!(vec, UInt16(i - 1))
    end

    n = length(pair.conf)
    key_to_unique = Dict{UInt64, UInt32}()
    sizehint!(key_to_unique, n)
    @inbounds for i in 1:n
        k = pack_pose_key(pair.conf[i], pair.rot[i], pair.offidx[i])
        haskey(key_to_unique, k) && error("new B (unique-K) contains duplicates")
        key_to_unique[k] = UInt32(UInt64(global_start) + UInt64(i - 1))
    end
    return BLookupSecondPass(pair.center, rel_to_indices, key_to_unique)
end

function match_active_first_pass!(
    active::ActivePool,
    b::BLookupFirstPass,
)::MatchResultFirstPass
    n = active_length(active)
    if n == 0
        return MatchResultFirstPass(UInt16[], UInt16[], Int16[], Int16[], Int16[], UInt32[])
    end

    valid = falses(n)
    rx = Vector{Int8}(undef, n)
    ry = Vector{Int8}(undef, n)
    rz = Vector{Int8}(undef, n)
    cx = Int32(b.center[1])
    cy = Int32(b.center[2])
    cz = Int32(b.center[3])

    @threads for i in 1:n
        dx = Int32(active.x[i]) - cx
        dy = Int32(active.y[i]) - cy
        dz = Int32(active.z[i]) - cz
        if dx >= I16_MIN && dx <= I16_MAX && dy >= I16_MIN && dy <= I16_MAX && dz >= I16_MIN && dz <= I16_MAX
            valid[i] = true
            rx[i] = Int8(dx)
            ry[i] = Int8(dy)
            rz[i] = Int8(dz)
        end
    end

    counts = copy(b.key_counts)
    matched = falses(n)
    m_conf = UInt16[]
    m_rot = UInt16[]
    m_x = Int16[]
    m_y = Int16[]
    m_z = Int16[]
    m_aidx = UInt32[]
    sizehint!(m_conf, n ÷ 4 + 1)
    sizehint!(m_rot, n ÷ 4 + 1)
    sizehint!(m_x, n ÷ 4 + 1)
    sizehint!(m_y, n ÷ 4 + 1)
    sizehint!(m_z, n ÷ 4 + 1)
    sizehint!(m_aidx, n ÷ 4 + 1)

    @inbounds for i in 1:n
        valid[i] || continue
        relkey = pack_rel_key(rx[i], ry[i], rz[i])
        idxs = get(b.rel_to_indices, relkey, nothing)
        idxs === nothing && continue
        c = active.conf[i]
        r = active.rot[i]
        ok = false
        for oi in idxs
            k = pack_pose_key(c, r, oi)
            rem = get(counts, k, Int32(0))
            if rem > 0
                counts[k] = rem - 1
                ok = true
                break
            end
        end
        ok || continue
        matched[i] = true
        push!(m_conf, c)
        push!(m_rot, r)
        push!(m_x, active.x[i])
        push!(m_y, active.y[i])
        push!(m_z, active.z[i])
        push!(m_aidx, active.aidx[i])
    end

    if any(matched)
        s_conf = UInt16[]
        s_rot = UInt16[]
        s_x = Int16[]
        s_y = Int16[]
        s_z = Int16[]
        s_aidx = UInt32[]
        sizehint!(s_conf, n - length(m_conf))
        sizehint!(s_rot, n - length(m_conf))
        sizehint!(s_x, n - length(m_conf))
        sizehint!(s_y, n - length(m_conf))
        sizehint!(s_z, n - length(m_conf))
        sizehint!(s_aidx, n - length(m_conf))
        @inbounds for i in 1:n
            matched[i] && continue
            push!(s_conf, active.conf[i])
            push!(s_rot, active.rot[i])
            push!(s_x, active.x[i])
            push!(s_y, active.y[i])
            push!(s_z, active.z[i])
            push!(s_aidx, active.aidx[i])
        end
        active.conf = s_conf
        active.rot = s_rot
        active.x = s_x
        active.y = s_y
        active.z = s_z
        active.aidx = s_aidx
    end

    return MatchResultFirstPass(m_conf, m_rot, m_x, m_y, m_z, m_aidx)
end

function match_active_second_pass!(
    active::ActivePool,
    b::BLookupSecondPass,
)::MatchResultSecondPass
    n = active_length(active)
    if n == 0
        return MatchResultSecondPass(UInt32[], UInt32[])
    end

    valid = falses(n)
    rx = Vector{Int8}(undef, n)
    ry = Vector{Int8}(undef, n)
    rz = Vector{Int8}(undef, n)
    cx = Int32(b.center[1])
    cy = Int32(b.center[2])
    cz = Int32(b.center[3])

    @threads for i in 1:n
        dx = Int32(active.x[i]) - cx
        dy = Int32(active.y[i]) - cy
        dz = Int32(active.z[i]) - cz
        if dx >= I16_MIN && dx <= I16_MAX && dy >= I16_MIN && dy <= I16_MAX && dz >= I16_MIN && dz <= I16_MAX
            valid[i] = true
            rx[i] = Int8(dx)
            ry[i] = Int8(dy)
            rz[i] = Int8(dz)
        end
    end

    matched = falses(n)
    m_aidx = UInt32[]
    m_uidx = UInt32[]
    sizehint!(m_aidx, n ÷ 4 + 1)
    sizehint!(m_uidx, n ÷ 4 + 1)

    @inbounds for i in 1:n
        valid[i] || continue
        relkey = pack_rel_key(rx[i], ry[i], rz[i])
        idxs = get(b.rel_to_indices, relkey, nothing)
        idxs === nothing && continue
        c = active.conf[i]
        r = active.rot[i]
        found = false
        found_uidx = UInt32(0)
        for oi in idxs
            u = get(b.key_to_unique, pack_pose_key(c, r, oi), UInt32(0xffffffff))
            if u != UInt32(0xffffffff)
                found = true
                found_uidx = u
                break
            end
        end
        found || continue
        matched[i] = true
        push!(m_aidx, active.aidx[i])
        push!(m_uidx, found_uidx)
    end

    if any(matched)
        s_conf = UInt16[]
        s_rot = UInt16[]
        s_x = Int16[]
        s_y = Int16[]
        s_z = Int16[]
        s_aidx = UInt32[]
        sizehint!(s_conf, n - length(m_aidx))
        sizehint!(s_rot, n - length(m_aidx))
        sizehint!(s_x, n - length(m_aidx))
        sizehint!(s_y, n - length(m_aidx))
        sizehint!(s_z, n - length(m_aidx))
        sizehint!(s_aidx, n - length(m_aidx))
        @inbounds for i in 1:n
            matched[i] && continue
            push!(s_conf, active.conf[i])
            push!(s_rot, active.rot[i])
            push!(s_x, active.x[i])
            push!(s_y, active.y[i])
            push!(s_z, active.z[i])
            push!(s_aidx, active.aidx[i])
        end
        active.conf = s_conf
        active.rot = s_rot
        active.x = s_x
        active.y = s_y
        active.z = s_z
        active.aidx = s_aidx
    end

    return MatchResultSecondPass(m_aidx, m_uidx)
end

function init_pose_writer(
    outdir::String;
    start_index::Int = 1,
    max_poses_per_chunk::Int = Int(typemax(UInt32)),
    write_empty_if_none::Bool = false,
)::PoseFileWriter
    mkpath(outdir)
    return PoseFileWriter(
        outdir,
        start_index,
        max_poses_per_chunk,
        write_empty_if_none,
        UInt64(0),
        UInt16[],
        UInt16[],
        UInt16[],
        Dict{NTuple{3, Int16}, UInt16}(),
        NTuple{3, Int16}[],
        nothing,
        nothing,
    )
end

function can_add_offset!(
    w::PoseFileWriter,
    xyz::NTuple{3, Int16},
)::Tuple{Bool, UInt16}
    existing = get(w.offset_map, xyz, UInt16(0xffff))
    if existing != UInt16(0xffff)
        return true, existing
    end
    length(w.offsets) < 65536 || return false, UInt16(0)

    x32 = Int32(xyz[1])
    y32 = Int32(xyz[2])
    z32 = Int32(xyz[3])
    if w.min_xyz === nothing
        new_min = (x32, y32, z32)
        new_max = (x32, y32, z32)
    else
        mn = w.min_xyz::NTuple{3, Int32}
        mx = w.max_xyz::NTuple{3, Int32}
        new_min = (min(mn[1], x32), min(mn[2], y32), min(mn[3], z32))
        new_max = (max(mx[1], x32), max(mx[2], y32), max(mx[3], z32))
    end
    if (new_max[1] - new_min[1]) > 255 || (new_max[2] - new_min[2]) > 255 || (new_max[3] - new_min[3]) > 255
        return false, UInt16(0)
    end

    idx = UInt16(length(w.offsets))
    w.offset_map[xyz] = idx
    push!(w.offsets, xyz)
    w.min_xyz = new_min
    w.max_xyz = new_max
    return true, idx
end

function flush_pose_writer!(w::PoseFileWriter)
    n = length(w.conf)
    if n == 0
        return
    end

    pose_path = joinpath(w.outdir, "poses-$(w.next_index).npy")
    offsets_path = joinpath(w.outdir, "offsets-$(w.next_index).dat")

    write_npy_u16_3cols(pose_path, w.conf, w.rot, w.offidx)

    noff = length(w.offsets)
    center = NTuple{3, Int16}((
        Int16(clamp(Int((w.max_xyz::NTuple{3, Int32})[1] - 127), typemin(Int16), typemax(Int16))),
        Int16(clamp(Int((w.max_xyz::NTuple{3, Int32})[2] - 127), typemin(Int16), typemax(Int16))),
        Int16(clamp(Int((w.max_xyz::NTuple{3, Int32})[3] - 127), typemin(Int16), typemax(Int16))),
    ))
    open(offsets_path, "w") do io
        write_le_u16(io, i16_to_u16(center[1]))
        write_le_u16(io, i16_to_u16(center[2]))
        write_le_u16(io, i16_to_u16(center[3]))
        @inbounds for i in 1:noff
            ox, oy, oz = w.offsets[i]
            dx = Int32(ox) - Int32(center[1])
            dy = Int32(oy) - Int32(center[2])
            dz = Int32(oz) - Int32(center[3])
            (dx >= -128 && dx <= 127 && dy >= -128 && dy <= 127 && dz >= -128 && dz <= 127) ||
                error("Offset cannot be encoded as int8 around center")
            write(io, i8_to_u8(Int8(dx)))
            write(io, i8_to_u8(Int8(dy)))
            write(io, i8_to_u8(Int8(dz)))
        end
    end

    w.total_written += UInt64(n)
    w.next_index += 1
    empty!(w.conf)
    empty!(w.rot)
    empty!(w.offidx)
    empty!(w.offset_map)
    empty!(w.offsets)
    w.min_xyz = nothing
    w.max_xyz = nothing
end

function add_pose!(
    w::PoseFileWriter,
    conf::UInt16,
    rot::UInt16,
    x::Int16,
    y::Int16,
    z::Int16,
)
    if length(w.conf) >= w.max_poses_per_chunk
        flush_pose_writer!(w)
    end

    xyz = (x, y, z)
    ok, idx = can_add_offset!(w, xyz)
    if !ok
        if !isempty(w.conf)
            flush_pose_writer!(w)
            ok2, idx2 = can_add_offset!(w, xyz)
            ok2 || error("Single pose violates chunk constraints")
            idx = idx2
        else
            error("Single pose violates chunk constraints")
        end
    end

    push!(w.conf, conf)
    push!(w.rot, rot)
    push!(w.offidx, idx)
end

function add_pose_batch!(
    w::PoseFileWriter,
    conf::Vector{UInt16},
    rot::Vector{UInt16},
    x::Vector{Int16},
    y::Vector{Int16},
    z::Vector{Int16},
)
    n = length(conf)
    length(rot) == n || error("Mismatched lengths in add_pose_batch!")
    length(x) == n || error("Mismatched lengths in add_pose_batch!")
    length(y) == n || error("Mismatched lengths in add_pose_batch!")
    length(z) == n || error("Mismatched lengths in add_pose_batch!")
    @inbounds for i in 1:n
        add_pose!(w, conf[i], rot[i], x[i], y[i], z[i])
    end
end

function finish_pose_writer!(w::PoseFileWriter)
    if !isempty(w.conf)
        flush_pose_writer!(w)
    end
    if w.total_written == 0 && w.write_empty_if_none
        pose_path = joinpath(w.outdir, "poses-$(w.next_index).npy")
        offsets_path = joinpath(w.outdir, "offsets-$(w.next_index).dat")
        write_npy_u16_3cols(pose_path, UInt16[], UInt16[], UInt16[])
        open(offsets_path, "w") do io
            write_le_u16(io, i16_to_u16(Int16(0)))
            write_le_u16(io, i16_to_u16(Int16(0)))
            write_le_u16(io, i16_to_u16(Int16(0)))
        end
        w.next_index += 1
    end
end

function init_index_writer(
    outdir::String,
    prefix::String;
    start_index::Int = 1,
    ncols::Int = 1,
    flush_rows::Int = 2_000_000,
)::IndexFileWriter
    mkpath(outdir)
    ncols == 1 || ncols == 2 || error("ncols must be 1 or 2")
    return IndexFileWriter(
        outdir,
        prefix,
        start_index,
        ncols,
        flush_rows,
        UInt64(0),
        UInt32[],
        UInt32[],
    )
end

function flush_index_writer!(w::IndexFileWriter)
    n = length(w.col1)
    n == 0 && return
    path = joinpath(w.outdir, "$(w.prefix)-$(w.next_index).npy")
    if w.ncols == 1
        write_npy_u32_1col(path, w.col1)
    else
        length(w.col2) == n || error("Index writer column length mismatch")
        write_npy_u32_2cols(path, w.col1, w.col2)
    end
    w.total_rows += UInt64(n)
    w.next_index += 1
    empty!(w.col1)
    empty!(w.col2)
end

function add_index1!(w::IndexFileWriter, v::Vector{UInt32})
    w.ncols == 1 || error("add_index1! requires ncols=1")
    append!(w.col1, v)
    if length(w.col1) >= w.flush_rows
        flush_index_writer!(w)
    end
end

function add_index2!(w::IndexFileWriter, c1::Vector{UInt32}, c2::Vector{UInt32})
    w.ncols == 2 || error("add_index2! requires ncols=2")
    length(c1) == length(c2) || error("add_index2! length mismatch")
    append!(w.col1, c1)
    append!(w.col2, c2)
    if length(w.col1) >= w.flush_rows
        flush_index_writer!(w)
    end
end

function finish_index_writer!(w::IndexFileWriter)
    flush_index_writer!(w)
end

function reserve_push!(
    reserve::Dict{Int, ActivePool},
    key::Int,
    active::ActivePool,
)
    ap = get!(reserve, key, init_active_pool())
    append_active!(ap, active)
end

function run_first_pass_to_key_buckets!(
    a_pairs::Vector{PairInfo},
    b_pairs::Vector{PairInfo},
    bucket_dir::String,
    cfg::CliConfig,
)::UInt64
    nb = cfg.dedup_buckets
    mkpath(bucket_dir)
    bucket_paths = [joinpath(bucket_dir, @sprintf("keybucket-%05d.bin", i)) for i in 0:(nb - 1)]
    bucket_ios = [open(p, "w") for p in bucket_paths]

    total_matches = UInt64(0)
    reserve = Dict{Int, ActivePool}()
    a_counter = UInt64(0)
    try
        for aa in a_pairs
            aa_loaded = load_pair(aa)
            active = init_active_pool()
            add_new_aa_to_active!(active, aa_loaded, a_counter)

            for (bb_i, bb) in enumerate(b_pairs)
                if haskey(reserve, bb_i)
                    append_active!(active, reserve[bb_i])
                    delete!(reserve, bb_i)
                end
                active_length(active) == 0 && continue

                bb_loaded = load_pair(bb)
                b_lookup = make_blookup_first(bb_loaded)
                matched = match_active_first_pass!(active, b_lookup)

                n = length(matched.conf)
                total_matches += UInt64(n)
                @inbounds for i in 1:n
                    bidx = key_bucket_id(
                        matched.conf[i],
                        matched.rot[i],
                        matched.x[i],
                        matched.y[i],
                        matched.z[i],
                        nb,
                    )
                    write_key_record(
                        bucket_ios[bidx],
                        KeyRecord(
                            matched.conf[i],
                            matched.rot[i],
                            matched.x[i],
                            matched.y[i],
                            matched.z[i],
                        ),
                    )
                end

                # Note 1: do not reserve/break on the last BB pair.
                if active_length(active) < cfg.threshold_m && bb_i < length(b_pairs)
                    reserve_push!(reserve, bb_i, active)
                    clear_active!(active)
                    break
                end
            end

            clear_active!(active)
            a_counter += UInt64(length(aa_loaded.conf))
        end
    finally
        for io in bucket_ios
            close(io)
        end
    end

    return total_matches
end

function dedup_key_buckets_to_unique_k!(
    bucket_dir::String,
    outdir::String,
    cfg::CliConfig,
)::UInt64
    nb = cfg.dedup_buckets
    unique_writer = init_pose_writer(
        outdir;
        start_index = 1,
        max_poses_per_chunk = cfg.max_poses_per_chunk,
        write_empty_if_none = true,
    )

    for b in 0:(nb - 1)
        path = joinpath(bucket_dir, @sprintf("keybucket-%05d.bin", b))
        isfile(path) || continue
        sz = stat(path).size
        sz == 0 && continue
        sz % 10 == 0 || error("Corrupt key bucket file size: $path")
        bytes = read(path)
        n = Int(sz ÷ 10)
        recs = Vector{KeyRecord}(undef, n)
        p = 1
        @inbounds for i in 1:n
            recs[i] = read_key_record(bytes, p)
            p += 10
        end

        sort!(recs, by = r -> (r.conf, r.rot, r.x, r.y, r.z))
        i = 1
        while i <= n
            r0 = recs[i]
            add_pose!(unique_writer, r0.conf, r0.rot, r0.x, r0.y, r0.z)
            key = (r0.conf, r0.rot, r0.x, r0.y, r0.z)
            i += 1
            while i <= n && (recs[i].conf, recs[i].rot, recs[i].x, recs[i].y, recs[i].z) == key
                i += 1
            end
        end
        rm(path; force = true)
    end
    finish_pose_writer!(unique_writer)
    return unique_writer.total_written
end

function run_index_pass!(
    a_pairs::Vector{PairInfo},
    b_pairs::Vector{PairInfo},
    outdir::String,
    cfg::CliConfig;
    prefix::String,
)
    starts = Vector{UInt32}(undef, length(b_pairs))
    cursor = UInt64(0)
    for (i, p) in enumerate(b_pairs)
        cursor <= UInt64(typemax(UInt32)) || error("Unique-K index exceeds uint32 range")
        starts[i] = UInt32(cursor)
        cursor += UInt64(pose_row_count(p.pose_path))
    end
    cursor <= UInt64(typemax(UInt32)) + UInt64(1) || error("Unique-K index exceeds uint32 range")

    writer = init_index_writer(outdir, prefix; start_index = 1, ncols = 2, flush_rows = cfg.threshold_k)
    buf_idx = UInt32[]
    buf_uidx = UInt32[]
    reserve = Dict{Int, ActivePool}()
    a_counter = UInt64(0)

    for aa in a_pairs
        aa_loaded = load_pair(aa)
        active = init_active_pool()
        add_new_aa_to_active!(active, aa_loaded, a_counter)

        for (bb_i, bb) in enumerate(b_pairs)
            if haskey(reserve, bb_i)
                append_active!(active, reserve[bb_i])
                delete!(reserve, bb_i)
            end
            active_length(active) == 0 && continue

            bb_loaded = load_pair(bb)
            b_lookup = make_blookup_second(bb_loaded, starts[bb_i])
            matched = match_active_second_pass!(active, b_lookup)

            if !isempty(matched.aidx)
                append!(buf_idx, matched.aidx)
                append!(buf_uidx, matched.uniq_idx)
            end

            if length(buf_idx) >= cfg.threshold_k
                add_index2!(writer, buf_idx, buf_uidx)
                empty!(buf_idx)
                empty!(buf_uidx)
            end

            # Note 1: do not reserve/break on the last BB pair.
            if active_length(active) < cfg.threshold_m && bb_i < length(b_pairs)
                reserve_push!(reserve, bb_i, active)
                clear_active!(active)
                break
            end
        end

        clear_active!(active)
        a_counter += UInt64(length(aa_loaded.conf))
    end

    if !isempty(buf_idx)
        add_index2!(writer, buf_idx, buf_uidx)
    end
    finish_index_writer!(writer)
end

function run_first_pass!(
    a_pairs::Vector{PairInfo},
    b_pairs::Vector{PairInfo},
    raw_k_dir::String,
    raw_ind_a_dir::String,
    cfg::CliConfig,
)::FirstPassResult
    k_writer = init_pose_writer(
        raw_k_dir;
        start_index = 1,
        max_poses_per_chunk = cfg.max_poses_per_chunk,
        write_empty_if_none = false,
    )
    ki_writer = init_index_writer(raw_ind_a_dir, "ind-A"; start_index = 1, ncols = 1, flush_rows = cfg.threshold_k)

    kbuf_conf = UInt16[]
    kbuf_rot = UInt16[]
    kbuf_x = Int16[]
    kbuf_y = Int16[]
    kbuf_z = Int16[]
    kbuf_aidx = UInt32[]

    reserve = Dict{Int, ActivePool}()
    a_counter = UInt64(0)
    flushed_any = false

    for aa in a_pairs
        aa_loaded = load_pair(aa)
        active = init_active_pool()
        add_new_aa_to_active!(active, aa_loaded, a_counter)

        for (bb_i, bb) in enumerate(b_pairs)
            if haskey(reserve, bb_i)
                append_active!(active, reserve[bb_i])
                delete!(reserve, bb_i)
            end
            active_length(active) == 0 && continue

            bb_loaded = load_pair(bb)
            b_lookup = make_blookup_first(bb_loaded)
            matched = match_active_first_pass!(active, b_lookup)

            if !isempty(matched.conf)
                append!(kbuf_conf, matched.conf)
                append!(kbuf_rot, matched.rot)
                append!(kbuf_x, matched.x)
                append!(kbuf_y, matched.y)
                append!(kbuf_z, matched.z)
                append!(kbuf_aidx, matched.aidx)
            end

            if length(kbuf_conf) >= cfg.threshold_k
                add_pose_batch!(k_writer, kbuf_conf, kbuf_rot, kbuf_x, kbuf_y, kbuf_z)
                add_index1!(ki_writer, kbuf_aidx)
                flushed_any = true
                empty!(kbuf_conf)
                empty!(kbuf_rot)
                empty!(kbuf_x)
                empty!(kbuf_y)
                empty!(kbuf_z)
                empty!(kbuf_aidx)
            end

            # Note 1: do not reserve/break on the last BB pair.
            if active_length(active) < cfg.threshold_m && bb_i < length(b_pairs)
                reserve_push!(reserve, bb_i, active)
                clear_active!(active)
                break
            end
        end

        clear_active!(active)
        a_counter += UInt64(length(aa_loaded.conf))
    end

    pending::Union{Nothing, PendingRawK} = nothing
    if !isempty(kbuf_conf) && flushed_any
        add_pose_batch!(k_writer, kbuf_conf, kbuf_rot, kbuf_x, kbuf_y, kbuf_z)
        add_index1!(ki_writer, kbuf_aidx)
    elseif !isempty(kbuf_conf)
        # Note 2: if K was never flushed before, keep K/KI in memory (no final flush).
        pending = PendingRawK(
            copy(kbuf_conf),
            copy(kbuf_rot),
            copy(kbuf_x),
            copy(kbuf_y),
            copy(kbuf_z),
            copy(kbuf_aidx),
        )
    end
    finish_pose_writer!(k_writer)
    finish_index_writer!(ki_writer)
    k_count = if pending === nothing
        k_writer.total_written
    else
        kp = pending::PendingRawK
        UInt64(length(kp.conf))
    end
    return FirstPassResult(k_count, pending)
end

function write_bucket_record(io::IO, r::BucketRecord)
    write_le_u16(io, r.conf)
    write_le_u16(io, r.rot)
    write_le_u16(io, i16_to_u16(r.x))
    write_le_u16(io, i16_to_u16(r.y))
    write_le_u16(io, i16_to_u16(r.z))
    write_le_u32(io, r.old_idx)
end

function write_key_record(io::IO, r::KeyRecord)
    write_le_u16(io, r.conf)
    write_le_u16(io, r.rot)
    write_le_u16(io, i16_to_u16(r.x))
    write_le_u16(io, i16_to_u16(r.y))
    write_le_u16(io, i16_to_u16(r.z))
end

function read_key_record(bytes::Vector{UInt8}, pos::Int)::KeyRecord
    conf = read_le_u16(bytes, pos)
    rot = read_le_u16(bytes, pos + 2)
    x = u16_to_i16(read_le_u16(bytes, pos + 4))
    y = u16_to_i16(read_le_u16(bytes, pos + 6))
    z = u16_to_i16(read_le_u16(bytes, pos + 8))
    return KeyRecord(conf, rot, x, y, z)
end

@inline function key_bucket_id(
    conf::UInt16,
    rot::UInt16,
    x::Int16,
    y::Int16,
    z::Int16,
    nbuckets::Int,
)::Int
    return Int(key_hash(conf, rot, x, y, z) & UInt64(nbuckets - 1)) + 1
end

function read_bucket_record(bytes::Vector{UInt8}, pos::Int)::BucketRecord
    conf = read_le_u16(bytes, pos)
    rot = read_le_u16(bytes, pos + 2)
    x = u16_to_i16(read_le_u16(bytes, pos + 4))
    y = u16_to_i16(read_le_u16(bytes, pos + 6))
    z = u16_to_i16(read_le_u16(bytes, pos + 8))
    old_idx = read_le_u32(bytes, pos + 10)
    return BucketRecord(conf, rot, x, y, z, old_idx)
end

function write_map_record(io::IO, r::MapRecord)
    write_le_u32(io, r.old_idx)
    write_le_u32(io, r.uniq_idx)
end

function read_map_record(bytes::Vector{UInt8}, pos::Int)::MapRecord
    old_idx = read_le_u32(bytes, pos)
    uniq_idx = read_le_u32(bytes, pos + 4)
    return MapRecord(old_idx, uniq_idx)
end

function key_hash(conf::UInt16, rot::UInt16, x::Int16, y::Int16, z::Int16)::UInt64
    h = UInt64(0x9e3779b97f4a7c15)
    h ⊻= UInt64(conf) * UInt64(0xbf58476d1ce4e5b9)
    h ⊻= UInt64(rot) * UInt64(0x94d049bb133111eb)
    h ⊻= UInt64(i16_to_u16(x)) * UInt64(0xd6e8feb86659fd93)
    h ⊻= UInt64(i16_to_u16(y)) * UInt64(0xa0761d6478bd642f)
    h ⊻= UInt64(i16_to_u16(z)) * UInt64(0xe7037ed1a0b428db)
    return h
end

function discover_prefix_npy(dir::String, prefix::String)::Vector{Tuple{Int, String}}
    files = Tuple{Int, String}[]
    for name in readdir(dir)
        startswith(name, "$prefix-") || continue
        endswith(name, ".npy") || continue
        text = name[(length(prefix) + 2):(end - 4)]
        all(isdigit, text) || continue
        idx = parse(Int, text)
        push!(files, (idx, joinpath(dir, name)))
    end
    sort!(files, by = x -> x[1])
    return files
end

function dedup_k_and_rewrite_ind_a!(
    raw_k_dir::String,
    raw_ind_a_dir::String,
    final_outdir::String,
    cfg::CliConfig,
    pending::Union{Nothing, PendingRawK},
)::UInt64
    bucket_dir = mktempdir(joinpath(final_outdir, "_tmp"); prefix = "k-buckets-")
    map_dir = mktempdir(joinpath(final_outdir, "_tmp"); prefix = "k-maps-")
    nb = cfg.dedup_buckets

    bucket_paths = [joinpath(bucket_dir, @sprintf("bucket-%05d.bin", i)) for i in 0:(nb - 1)]
    bucket_ios = [open(p, "w") for p in bucket_paths]

    old_k_count = UInt64(0)
    if pending !== nothing
        kp = pending::PendingRawK
        n = length(kp.conf)
        length(kp.rot) == n || error("Pending K shape mismatch")
        length(kp.x) == n || error("Pending K shape mismatch")
        length(kp.y) == n || error("Pending K shape mismatch")
        length(kp.z) == n || error("Pending K shape mismatch")
        length(kp.aidx) == n || error("Pending KI shape mismatch")
        @inbounds for i in 1:n
            old_k_count <= UInt64(typemax(UInt32)) || error("K index exceeds uint32 range")
            rec = BucketRecord(
                kp.conf[i],
                kp.rot[i],
                kp.x[i],
                kp.y[i],
                kp.z[i],
                UInt32(old_k_count),
            )
            b = Int(key_hash(rec.conf, rec.rot, rec.x, rec.y, rec.z) & UInt64(nb - 1)) + 1
            write_bucket_record(bucket_ios[b], rec)
            old_k_count += 1
        end
    else
        raw_k_pairs = PairInfo[]
        try
            raw_k_pairs = discover_pose_pairs(raw_k_dir)
        catch err
            if !occursin("No pose files found", sprint(showerror, err))
                rethrow(err)
            end
        end
        for pair in raw_k_pairs
            lp = load_pair(pair)
            n = length(lp.conf)
            @inbounds for i in 1:n
                old_k_count <= UInt64(typemax(UInt32)) || error("K index exceeds uint32 range")
                oi = Int(lp.offidx[i]) + 1
                rec = BucketRecord(
                    lp.conf[i],
                    lp.rot[i],
                    lp.abs_offsets[oi, 1],
                    lp.abs_offsets[oi, 2],
                    lp.abs_offsets[oi, 3],
                    UInt32(old_k_count),
                )
                b = Int(key_hash(rec.conf, rec.rot, rec.x, rec.y, rec.z) & UInt64(nb - 1)) + 1
                write_bucket_record(bucket_ios[b], rec)
                old_k_count += 1
            end
        end
    end
    for io in bucket_ios
        close(io)
    end

    unique_writer = init_pose_writer(
        final_outdir;
        start_index = 1,
        max_poses_per_chunk = cfg.max_poses_per_chunk,
        write_empty_if_none = true,
    )
    next_unique = UInt32(0)
    map_paths = String[]

    for b in 1:nb
        path = bucket_paths[b]
        isfile(path) || continue
        sz = stat(path).size
        sz == 0 && continue
        sz % 14 == 0 || error("Corrupt bucket file size: $path")
        bytes = read(path)
        n = Int(sz ÷ 14)
        recs = Vector{BucketRecord}(undef, n)
        p = 1
        @inbounds for i in 1:n
            recs[i] = read_bucket_record(bytes, p)
            p += 14
        end

        sort!(recs, by = r -> (r.conf, r.rot, r.x, r.y, r.z, r.old_idx))
        maps = MapRecord[]
        sizehint!(maps, n)

        i = 1
        while i <= n
            r0 = recs[i]
            key = (r0.conf, r0.rot, r0.x, r0.y, r0.z)
            uid = next_unique
            add_pose!(unique_writer, r0.conf, r0.rot, r0.x, r0.y, r0.z)
            next_unique += UInt32(1)
            while i <= n
                r = recs[i]
                if (r.conf, r.rot, r.x, r.y, r.z) != key
                    break
                end
                push!(maps, MapRecord(r.old_idx, uid))
                i += 1
            end
        end

        sort!(maps, by = m -> m.old_idx)
        map_path = joinpath(map_dir, @sprintf("map-%05d.bin", b - 1))
        open(map_path, "w") do io
            @inbounds for m in maps
                write_map_record(io, m)
            end
        end
        push!(map_paths, map_path)
    end
    finish_pose_writer!(unique_writer)

    function heap_push!(heap::Vector{HeapNode}, node::HeapNode)
        push!(heap, node)
        i = length(heap)
        while i > 1
            p = i >>> 1
            if heap[p].old_idx <= heap[i].old_idx
                break
            end
            heap[p], heap[i] = heap[i], heap[p]
            i = p
        end
    end

    function heap_pop!(heap::Vector{HeapNode})::HeapNode
        isempty(heap) && error("heap_pop! on empty heap")
        out = heap[1]
        last = pop!(heap)
        if !isempty(heap)
            heap[1] = last
            i = 1
            while true
                l = i << 1
                r = l + 1
                smallest = i
                if l <= length(heap) && heap[l].old_idx < heap[smallest].old_idx
                    smallest = l
                end
                if r <= length(heap) && heap[r].old_idx < heap[smallest].old_idx
                    smallest = r
                end
                smallest == i && break
                heap[i], heap[smallest] = heap[smallest], heap[i]
                i = smallest
            end
        end
        return out
    end

    map_ios = IO[]
    for p in map_paths
        push!(map_ios, open(p, "r"))
    end
    heap = HeapNode[]
    for (src, io) in enumerate(map_ios)
        eof(io) && continue
        buf = read(io, 8)
        length(buf) == 8 || error("Corrupt map record in $(map_paths[src])")
        rec = read_map_record(buf, 1)
        heap_push!(heap, HeapNode(rec.old_idx, rec.uniq_idx, src))
    end

    merged_uidx_path = joinpath(map_dir, "old-to-uniq.bin")
    open(merged_uidx_path, "w") do out
        expect = UInt64(0)
        while !isempty(heap)
            node = heap_pop!(heap)
            UInt64(node.old_idx) == expect || error("Map merge is not contiguous at old index $expect")
            write_le_u32(out, node.uniq_idx)
            expect += 1

            io = map_ios[node.src]
            if !eof(io)
                buf = read(io, 8)
                length(buf) == 8 || error("Corrupt map record in $(map_paths[node.src])")
                rec = read_map_record(buf, 1)
                heap_push!(heap, HeapNode(rec.old_idx, rec.uniq_idx, node.src))
            end
        end
        expect == old_k_count || error("Missing map rows: expected $old_k_count, saw $expect")
    end
    for io in map_ios
        close(io)
    end

    ind_a_writer = init_index_writer(final_outdir, "ind-A"; start_index = 1, ncols = 2, flush_rows = cfg.threshold_k)
    merged_io = open(merged_uidx_path, "r")
    produced = UInt64(0)
    if pending !== nothing
        kp = pending::PendingRawK
        n = length(kp.aidx)
        col1 = copy(kp.aidx)
        col2 = Vector{UInt32}(undef, n)
        @inbounds for i in 1:n
            eof(merged_io) && error("Not enough old->unique map entries")
            b = read(merged_io, 4)
            length(b) == 4 || error("Corrupt old-to-uniq mapping")
            col2[i] = read_le_u32(b, 1)
            produced += 1
        end
        add_index2!(ind_a_writer, col1, col2)
    else
        raw_ind_files = discover_prefix_npy(raw_ind_a_dir, "ind-A")
        for (_, path) in raw_ind_files
            arr = read_npy_u32(path)
            ndims(arr) == 2 || error("Expected 2D ind-A raw file: $path")
            size(arr, 2) == 1 || error("Expected ind-A raw with one column: $path")
            n = size(arr, 1)
            col1 = Vector{UInt32}(undef, n)
            col2 = Vector{UInt32}(undef, n)
            @inbounds for i in 1:n
                col1[i] = arr[i, 1]
                eof(merged_io) && error("Not enough old->unique map entries")
                b = read(merged_io, 4)
                length(b) == 4 || error("Corrupt old-to-uniq mapping")
                col2[i] = read_le_u32(b, 1)
                produced += 1
            end
            add_index2!(ind_a_writer, col1, col2)
        end
    end
    eof(merged_io) || error("old-to-uniq mapping has extra entries")
    close(merged_io)
    finish_index_writer!(ind_a_writer)
    produced == old_k_count || error("ind-A row count mismatch with K rows")

    rm(bucket_dir; recursive = true, force = true)
    rm(map_dir; recursive = true, force = true)
    return unique_writer.total_written
end

function run_second_pass!(
    a_pairs::Vector{PairInfo},
    b_pairs::Vector{PairInfo},
    outdir::String,
    cfg::CliConfig,
)
    starts = Vector{UInt32}(undef, length(b_pairs))
    cursor = UInt64(0)
    for (i, p) in enumerate(b_pairs)
        cursor <= UInt64(typemax(UInt32)) || error("Unique-K index exceeds uint32 range")
        starts[i] = UInt32(cursor)
        cursor += UInt64(pose_row_count(p.pose_path))
    end
    cursor <= UInt64(typemax(UInt32)) + UInt64(1) || error("Unique-K index exceeds uint32 range")

    ib_writer = init_index_writer(outdir, "ind-B"; start_index = 1, ncols = 2, flush_rows = cfg.threshold_k)
    buf_bidx = UInt32[]
    buf_uidx = UInt32[]
    reserve = Dict{Int, ActivePool}()
    a_counter = UInt64(0)

    for aa in a_pairs
        aa_loaded = load_pair(aa)
        active = init_active_pool()
        add_new_aa_to_active!(active, aa_loaded, a_counter)

        for (bb_i, bb) in enumerate(b_pairs)
            if haskey(reserve, bb_i)
                append_active!(active, reserve[bb_i])
                delete!(reserve, bb_i)
            end
            active_length(active) == 0 && continue

            bb_loaded = load_pair(bb)
            b_lookup = make_blookup_second(bb_loaded, starts[bb_i])
            matched = match_active_second_pass!(active, b_lookup)

            if !isempty(matched.aidx)
                append!(buf_bidx, matched.aidx)
                append!(buf_uidx, matched.uniq_idx)
            end

            if length(buf_bidx) >= cfg.threshold_k
                add_index2!(ib_writer, buf_bidx, buf_uidx)
                empty!(buf_bidx)
                empty!(buf_uidx)
            end

            # Note 1: do not reserve/break on the last BB pair.
            if active_length(active) < cfg.threshold_m && bb_i < length(b_pairs)
                reserve_push!(reserve, bb_i, active)
                clear_active!(active)
                break
            end
        end

        clear_active!(active)
        a_counter += UInt64(length(aa_loaded.conf))
    end

    if !isempty(buf_bidx)
        add_index2!(ib_writer, buf_bidx, buf_uidx)
    end
    finish_index_writer!(ib_writer)
end

function sum_pose_file_sizes(pairs::Vector{PairInfo})::Int64
    s = Int64(0)
    for p in pairs
        s += p.pose_file_size
    end
    return s
end

function dir_size_bytes(path::String)::UInt64
    isdir(path) || return UInt64(0)
    total = UInt64(0)
    for (root, _, files) in walkdir(path)
        for f in files
            fp = joinpath(root, f)
            try
                total += UInt64(stat(fp).size)
            catch
                # Ignore files that disappear during cleanup.
            end
        end
    end
    return total
end

function fmt_bytes(n::UInt64)::String
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    v = Float64(n)
    i = 1
    while v >= 1024.0 && i < length(units)
        v /= 1024.0
        i += 1
    end
    return @sprintf("%.2f %s", v, units[i])
end

function main(argv::Vector{String})::Int
    cfg = parse_args(argv)
    isdir(cfg.set1) || error("Input directory not found: $(cfg.set1)")
    isdir(cfg.set2) || error("Input directory not found: $(cfg.set2)")
    !ispath(cfg.output) || error("Output path already exists: $(cfg.output)")

    pairs1 = discover_pose_pairs(cfg.set1)
    pairs2 = discover_pose_pairs(cfg.set2)
    size1 = sum_pose_file_sizes(pairs1)
    size2 = sum_pose_file_sizes(pairs2)

    large_pairs, small_pairs, large_name, small_name = if size1 >= size2
        (pairs1, pairs2, cfg.set1, cfg.set2)
    else
        (pairs2, pairs1, cfg.set2, cfg.set1)
    end

    mkpath(cfg.output)
    tmp_root = joinpath(cfg.output, "_tmp")
    mkpath(tmp_root)
    key_bucket_dir = joinpath(tmp_root, "key-buckets")
    mkpath(key_bucket_dir)

    println("A (largest): $large_name")
    println("B (smallest): $small_name")
    println("threads: $(nthreads())")

    first_pass_count = run_first_pass_to_key_buckets!(large_pairs, small_pairs, key_bucket_dir, cfg)
    println("first-pass K rows: $first_pass_count")
    println("tmp size after first pass: $(fmt_bytes(dir_size_bytes(tmp_root)))")

    unique_k_count = dedup_key_buckets_to_unique_k!(key_bucket_dir, cfg.output, cfg)
    println("unique-K rows: $unique_k_count")
    println("tmp size after dedup: $(fmt_bytes(dir_size_bytes(tmp_root)))")

    unique_k_pairs = discover_pose_pairs(cfg.output)
    run_index_pass!(large_pairs, unique_k_pairs, cfg.output, cfg; prefix = "ind-A")
    println("tmp size after ind-A: $(fmt_bytes(dir_size_bytes(tmp_root)))")
    run_index_pass!(small_pairs, unique_k_pairs, cfg.output, cfg; prefix = "ind-B")
    println("tmp size after ind-B: $(fmt_bytes(dir_size_bytes(tmp_root)))")
    println("done")
    println("output size: $(fmt_bytes(dir_size_bytes(cfg.output)))")

    if cfg.keep_temp
        println("kept temp directory: $tmp_root")
    else
        rm(tmp_root; recursive = true, force = true)
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(copy(ARGS)))
end
