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
    reuse_unique_k_dir::Union{Nothing, String}
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

struct PairWithStart
    pair::PairInfo
end

struct KeyIdxRecord
    key::UInt64
    idx::UInt32
end

mutable struct CanonicalSetIndex
    pairs_by_center::Dict{NTuple{3, Int16}, Vector{PairWithStart}}
    bucket_by_center::Dict{NTuple{3, Int16}, Vector{String}}
    total_poses::UInt64
    temp_bytes::UInt64
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

mutable struct Index64FileWriter
    outdir::String
    prefix::String
    next_index::Int
    flush_rows::Int
    total_rows::UInt64
    col1::Vector{UInt64}
    col2::Vector{UInt64}
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
          --reuse-unique-k-dir <dir>  Skip prefilter/canonicalization/unique-k and reuse existing unique-K poses
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
    reuse_unique_k_dir = nothing

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
        elseif arg == "--reuse-unique-k-dir"
            i + 1 <= length(argv) || error("--reuse-unique-k-dir requires a value")
            reuse_unique_k_dir = argv[i + 1]
            i += 2
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
        reuse_unique_k_dir,
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

function read_le_u64(bytes::Vector{UInt8}, pos::Int)::UInt64
    return (
        UInt64(bytes[pos]) |
        (UInt64(bytes[pos + 1]) << 8) |
        (UInt64(bytes[pos + 2]) << 16) |
        (UInt64(bytes[pos + 3]) << 24) |
        (UInt64(bytes[pos + 4]) << 32) |
        (UInt64(bytes[pos + 5]) << 40) |
        (UInt64(bytes[pos + 6]) << 48) |
        (UInt64(bytes[pos + 7]) << 56)
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

function write_le_u64(io::IO, x::UInt64)
    write(io, UInt8(x & 0x00000000000000ff))
    write(io, UInt8((x >> 8) & 0x00000000000000ff))
    write(io, UInt8((x >> 16) & 0x00000000000000ff))
    write(io, UInt8((x >> 24) & 0x00000000000000ff))
    write(io, UInt8((x >> 32) & 0x00000000000000ff))
    write(io, UInt8((x >> 40) & 0x00000000000000ff))
    write(io, UInt8((x >> 48) & 0x00000000000000ff))
    write(io, UInt8((x >> 56) & 0x00000000000000ff))
end

@inline function write_u16_vector_le(io::IO, data::Vector{UInt16})
    if ENDIAN_BOM == 0x04030201
        write(io, reinterpret(UInt8, data))
    else
        @inbounds for v in data
            write_le_u16(io, v)
        end
    end
end

@inline function write_u64_vector_le(io::IO, data::Vector{UInt64})
    if ENDIAN_BOM == 0x04030201
        write(io, reinterpret(UInt8, data))
    else
        @inbounds for v in data
            write_le_u64(io, v)
        end
    end
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

@inline function pack_rel_pose_key(
    conf::UInt16,
    rot::UInt16,
    rx::Int8,
    ry::Int8,
    rz::Int8,
)::UInt64
    return (
        UInt64(conf) |
        (UInt64(rot) << 16) |
        (UInt64(reinterpret(UInt8, rx)) << 32) |
        (UInt64(reinterpret(UInt8, ry)) << 40) |
        (UInt64(reinterpret(UInt8, rz)) << 48)
    )
end

@inline function pack_rel_suffix(rx::Int8, ry::Int8, rz::Int8)::UInt64
    return (
        (UInt64(reinterpret(UInt8, rx)) << 32) |
        (UInt64(reinterpret(UInt8, ry)) << 40) |
        (UInt64(reinterpret(UInt8, rz)) << 48)
    )
end

@inline function unpack_rel_pose_key(key::UInt64)
    conf = UInt16(key & 0x0000_0000_0000_ffff)
    rot = UInt16((key >> 16) & 0x0000_0000_0000_ffff)
    rx = reinterpret(Int8, UInt8((key >> 32) & 0x0000_0000_0000_00ff))
    ry = reinterpret(Int8, UInt8((key >> 40) & 0x0000_0000_0000_00ff))
    rz = reinterpret(Int8, UInt8((key >> 48) & 0x0000_0000_0000_00ff))
    return conf, rot, rx, ry, rz
end

@inline function is_center_canonical(c::NTuple{3, Int16})::Bool
    return mod(Int32(c[1]), 256) == 0 && mod(Int32(c[2]), 256) == 0 && mod(Int32(c[3]), 256) == 0
end

@inline function canonical_center_coord(v::Int16)::Int16
    c32 = fld(Int32(v) + 128, 256) * 256
    (c32 >= typemin(Int16) && c32 <= typemax(Int16)) || error("Canonical center out of int16 range for coordinate $v")
    return Int16(c32)
end

@inline function canonical_center(x::Int16, y::Int16, z::Int16)::NTuple{3, Int16}
    return (canonical_center_coord(x), canonical_center_coord(y), canonical_center_coord(z))
end

function dtype_descr(::Type{UInt16})::String
    return "<u2"
end

function dtype_descr(::Type{UInt32})::String
    return "<u4"
end

function dtype_descr(::Type{UInt64})::String
    return "<u8"
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

function write_npy_u64(path::String, arr::AbstractArray{UInt64})
    open(path, "w") do io
        if ndims(arr) == 1
            write_npy_header(io, dtype_descr(UInt64), [length(arr)])
            @inbounds for v in arr
                write_le_u64(io, v)
            end
        elseif ndims(arr) == 2
            nrows, ncols = size(arr)
            write_npy_header(io, dtype_descr(UInt64), [nrows, ncols])
            @inbounds for i in 1:nrows
                for j in 1:ncols
                    write_le_u64(io, arr[i, j])
                end
            end
        else
            error("write_npy_u64 supports only rank-1 or rank-2 arrays")
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
        write_u16_vector_le(io, raw)
    end
end

function write_npy_u16_3cols_zst(
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

    zstd_threads = zstd_thread_count()
    cmd = zstd_threads == 0 ?
          `zstd -q -f -T0 -o $path --` :
          `zstd -q -f -T$(zstd_threads) -o $path --`
    try
        open(cmd, "w") do io
            write_npy_header(io, dtype_descr(UInt16), [n, 3])
            write_u16_vector_le(io, raw)
        end
    catch
        # Fallback for zstd builds/options that reject -T on this path.
        open(`zstd -q -f -o $path --`, "w") do io
            write_npy_header(io, dtype_descr(UInt16), [n, 3])
            write_u16_vector_le(io, raw)
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

function write_npy_u64_2cols(path::String, col1::Vector{UInt64}, col2::Vector{UInt64})
    n = length(col1)
    length(col2) == n || error("Column length mismatch")
    raw = Vector{UInt64}(undef, n * 2)
    @inbounds for i in 1:n
        j = 2 * (i - 1)
        raw[j + 1] = col1[i]
        raw[j + 2] = col2[i]
    end
    open(path, "w") do io
        write_npy_header(io, dtype_descr(UInt64), [n, 2])
        @inbounds for v in raw
            write_le_u64(io, v)
        end
    end
end

function zstd_thread_count()::Int
    zstd_threads = try
        parse(Int, get(ENV, "CROCODILE_ZSTD_THREADS", "0"))
    catch
        0
    end
    return max(0, zstd_threads)
end

function decompress_zst_bytes(path::String; single_thread::Bool = false)::Vector{UInt8}
    zstd_threads = single_thread ? 1 : zstd_thread_count()
    cmd = zstd_threads == 0 ?
          `zstd -dc -T0 --force -- $path` :
          `zstd -dc -T$(zstd_threads) --force -- $path`
    try
        return read(cmd)
    catch err
        # Fallback for zstd builds/options that reject -T on this path.
        try
            return read(`zstd -dc --force -- $path`)
        catch err2
            error("Failed to read $path with zstd -dc: $err2 (threaded attempt: $err)")
        end
    end
end

function decompress_npy_zst(path::String)::Vector{UInt8}
    return decompress_zst_bytes(path)
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

function pose_row_count_zst(path::String)::Int
    descr, fortran, shape = try
        zstd_threads = zstd_thread_count()
        cmd = zstd_threads == 0 ?
              `zstd -dc -T0 --force -- $path` :
              `zstd -dc -T$(zstd_threads) --force -- $path`
        open(cmd, "r") do io
            parse_npy_header_io(io)
        end
    catch
        # Some zstd/Julia combinations can surface EPIPE when the stream is
        # closed early after reading only the header. Fall back to full decode.
        bytes = decompress_npy_zst(path)
        d, f, s, _ = parse_npy_header(bytes)
        (d, f, s)
    end
    (descr == "<u2" || descr == "|u2") || error("Expected uint16 .npy in $path, got $descr")
    fortran && error("Fortran-order arrays are not supported in $path")
    length(shape) == 2 || error("Expected 2D pose array in $path")
    shape[2] == 3 || error("Expected pose shape [N,3] in $path")
    return shape[1]
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
    elseif endswith(path, ".npy.zst")
        return pose_row_count_zst(path)
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

function read_offset_center(path::String)::NTuple{3, Int16}
    bytes = read(path)
    length(bytes) >= 6 || error("offset file too small: $path")
    return (
        u16_to_i16(read_le_u16(bytes, 1)),
        u16_to_i16(read_le_u16(bytes, 3)),
        u16_to_i16(read_le_u16(bytes, 5)),
    )
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

    pose_path = joinpath(w.outdir, "poses-$(w.next_index).npy.zst")
    offsets_path = joinpath(w.outdir, "offsets-$(w.next_index).dat")

    write_npy_u16_3cols_zst(pose_path, w.conf, w.rot, w.offidx)

    noff = length(w.offsets)
    center = NTuple{3, Int16}((
        Int16(clamp(Int((w.max_xyz::NTuple{3, Int32})[1] - 127), typemin(Int16), typemax(Int16))),
        Int16(clamp(Int((w.max_xyz::NTuple{3, Int32})[2] - 127), typemin(Int16), typemax(Int16))),
        Int16(clamp(Int((w.max_xyz::NTuple{3, Int32})[3] - 127), typemin(Int16), typemax(Int16))),
    ))
    offset_bytes = Vector{UInt8}(undef, 6 + 3 * noff)
    c1 = i16_to_u16(center[1])
    c2 = i16_to_u16(center[2])
    c3 = i16_to_u16(center[3])
    offset_bytes[1] = UInt8(c1 & 0x00ff)
    offset_bytes[2] = UInt8((c1 >> 8) & 0x00ff)
    offset_bytes[3] = UInt8(c2 & 0x00ff)
    offset_bytes[4] = UInt8((c2 >> 8) & 0x00ff)
    offset_bytes[5] = UInt8(c3 & 0x00ff)
    offset_bytes[6] = UInt8((c3 >> 8) & 0x00ff)
    p = 7
    @inbounds for i in 1:noff
        ox, oy, oz = w.offsets[i]
        dx = Int32(ox) - Int32(center[1])
        dy = Int32(oy) - Int32(center[2])
        dz = Int32(oz) - Int32(center[3])
        (dx >= -128 && dx <= 127 && dy >= -128 && dy <= 127 && dz >= -128 && dz <= 127) ||
            error("Offset cannot be encoded as int8 around center")
        offset_bytes[p] = i8_to_u8(Int8(dx))
        offset_bytes[p + 1] = i8_to_u8(Int8(dy))
        offset_bytes[p + 2] = i8_to_u8(Int8(dz))
        p += 3
    end
    open(offsets_path, "w") do io
        write(io, offset_bytes)
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
        pose_path = joinpath(w.outdir, "poses-$(w.next_index).npy.zst")
        offsets_path = joinpath(w.outdir, "offsets-$(w.next_index).dat")
        write_npy_u16_3cols_zst(pose_path, UInt16[], UInt16[], UInt16[])
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

function init_index64_writer(
    outdir::String,
    prefix::String;
    start_index::Int = 1,
    flush_rows::Int = 2_000_000,
)::Index64FileWriter
    mkpath(outdir)
    return Index64FileWriter(
        outdir,
        prefix,
        start_index,
        flush_rows,
        UInt64(0),
        UInt64[],
        UInt64[],
    )
end

function flush_index64_writer!(w::Index64FileWriter)
    n = length(w.col1)
    n == 0 && return
    length(w.col2) == n || error("Index64 writer column length mismatch")
    path = joinpath(w.outdir, "$(w.prefix)-$(w.next_index).npy")
    write_npy_u64_2cols(path, w.col1, w.col2)
    w.total_rows += UInt64(n)
    w.next_index += 1
    empty!(w.col1)
    empty!(w.col2)
end

function add_index64_2!(w::Index64FileWriter, c1::Vector{UInt64}, c2::Vector{UInt64})
    length(c1) == length(c2) || error("add_index64_2! length mismatch")
    append!(w.col1, c1)
    append!(w.col2, c2)
    if length(w.col1) >= w.flush_rows
        flush_index64_writer!(w)
    end
end

function finish_index64_writer!(w::Index64FileWriter)
    flush_index64_writer!(w)
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

@inline function center_key_filename(center::NTuple{3, Int16})::String
    return @sprintf("center-%d-%d-%d.bin", Int(center[1]), Int(center[2]), Int(center[3]))
end

@inline function write_keyidx_record(io::IO, rec::KeyIdxRecord)
    write_le_u64(io, rec.key)
    write_le_u32(io, rec.idx)
end

function read_keyidx_records(path::String)::Vector{KeyIdxRecord}
    isfile(path) || return KeyIdxRecord[]
    bytes = if endswith(path, ".zst")
        decompress_zst_bytes(path)
    else
        read(path)
    end
    sz = length(bytes)
    sz == 0 && return KeyIdxRecord[]
    sz % 12 == 0 || error("Corrupt keyidx file size: $path")
    n = Int(sz ÷ 12)
    out = Vector{KeyIdxRecord}(undef, n)
    p = 1
    @inbounds for i in 1:n
        out[i] = KeyIdxRecord(read_le_u64(bytes, p), read_le_u32(bytes, p + 8))
        p += 12
    end
    return out
end

function read_key_records(path::String; single_thread::Bool = false)::Vector{UInt64}
    isfile(path) || return UInt64[]
    bytes = if endswith(path, ".zst")
        decompress_zst_bytes(path; single_thread = single_thread)
    else
        read(path)
    end
    sz = length(bytes)
    sz == 0 && return UInt64[]
    sz % 8 == 0 || error("Corrupt key file size: $path")
    raw = reinterpret(UInt64, bytes)
    if ENDIAN_BOM == 0x04030201
        return collect(raw)
    end
    n = length(raw)
    out = Vector{UInt64}(undef, n)
    @inbounds for i in 1:n
        out[i] = bswap(raw[i])
    end
    return out
end

function open_zstd_writer(f::Function, path::String; single_thread::Bool = false)
    zstd_threads = single_thread ? 1 : zstd_thread_count()
    cmd = zstd_threads == 0 ?
          `zstd -q -f -T0 -o $path --` :
          `zstd -q -f -T$(zstd_threads) -o $path --`
    try
        open(cmd, "w") do io
            f(io)
        end
    catch
        # Fallback for zstd builds/options that reject -T on this path.
        open(`zstd -q -f -o $path --`, "w") do io
            f(io)
        end
    end
end

function zstd_single_thread_below_bytes()::Int
    try
        # Below this payload size, zstd threading overhead usually dominates.
        return max(1, parse(Int, get(ENV, "CROCODILE_ZSTD_SINGLE_THREAD_BELOW_BYTES", "67108864")))
    catch
        return 67_108_864
    end
end

@inline function zstd_should_use_single_thread(nbytes::Int)::Bool
    nthreads() <= 2 && return true
    nbytes < zstd_single_thread_below_bytes()
end

function write_u64_records_zst(path::String, values::Vector{UInt64}; single_thread::Bool = false)
    open_zstd_writer(path; single_thread = single_thread) do io
        write_u64_vector_le(io, values)
    end
end

function write_u64_records_raw(path::String, values::Vector{UInt64})
    open(path, "w") do io
        write_u64_vector_le(io, values)
    end
end

@inline function write_keyuid_record(io::IO, key::UInt64, uid::UInt32)
    write_le_u64(io, key)
    write_le_u32(io, uid)
end

@inline function write_keyuid_to_bytes!(buf::Vector{UInt8}, pos::Int, key::UInt64, uid::UInt32)
    @inbounds begin
        buf[pos] = UInt8(key & 0x00000000000000ff)
        buf[pos + 1] = UInt8((key >> 8) & 0x00000000000000ff)
        buf[pos + 2] = UInt8((key >> 16) & 0x00000000000000ff)
        buf[pos + 3] = UInt8((key >> 24) & 0x00000000000000ff)
        buf[pos + 4] = UInt8((key >> 32) & 0x00000000000000ff)
        buf[pos + 5] = UInt8((key >> 40) & 0x00000000000000ff)
        buf[pos + 6] = UInt8((key >> 48) & 0x00000000000000ff)
        buf[pos + 7] = UInt8((key >> 56) & 0x00000000000000ff)
        buf[pos + 8] = UInt8(uid & 0x000000ff)
        buf[pos + 9] = UInt8((uid >> 8) & 0x000000ff)
        buf[pos + 10] = UInt8((uid >> 16) & 0x000000ff)
        buf[pos + 11] = UInt8((uid >> 24) & 0x000000ff)
    end
end

function read_keyuid_records(path::String; single_thread::Bool = false)::Tuple{Vector{UInt64}, Vector{UInt32}}
    isfile(path) || return UInt64[], UInt32[]
    bytes = if endswith(path, ".zst")
        decompress_zst_bytes(path; single_thread = single_thread)
    else
        read(path)
    end
    sz = length(bytes)
    sz == 0 && return UInt64[], UInt32[]
    sz % 12 == 0 || error("Corrupt keyuid file size: $path")
    n = Int(sz ÷ 12)
    keys = Vector{UInt64}(undef, n)
    uids = Vector{UInt32}(undef, n)
    p = 1
    @inbounds for i in 1:n
        keys[i] = read_le_u64(bytes, p)
        uids[i] = read_le_u32(bytes, p + 8)
        p += 12
    end
    return keys, uids
end

function read_key_records10(path::String; single_thread::Bool = false)::Vector{KeyRecord}
    isfile(path) || return KeyRecord[]
    bytes = if endswith(path, ".zst")
        decompress_zst_bytes(path; single_thread = single_thread)
    else
        read(path)
    end
    sz = length(bytes)
    sz == 0 && return KeyRecord[]
    sz % 10 == 0 || error("Corrupt key-record file size: $path")
    n = Int(sz ÷ 10)
    out = Vector{KeyRecord}(undef, n)
    p = 1
    @inbounds for i in 1:n
        out[i] = read_key_record(bytes, p)
        p += 10
    end
    return out
end

function compress_file_zst_inplace(path::String)::String
    isfile(path) || return path
    endswith(path, ".zst") && return path
    out = path * ".zst"
    zstd_threads = zstd_thread_count()
    cmd = zstd_threads == 0 ?
          `zstd -q -f -T0 --rm -o $out -- $path` :
          `zstd -q -f -T$(zstd_threads) --rm -o $out -- $path`
    try
        run(cmd)
    catch
        # Fallback for zstd builds/options that reject -T on this path.
        run(`zstd -q -f --rm -o $out -- $path`)
    end
    return out
end

function compress_bytes_zst(raw::Vector{UInt8})::Vector{UInt8}
    zstd_threads = zstd_thread_count()
    cmd = zstd_threads == 0 ? `zstd -q -c -T0 --` : `zstd -q -c -T$(zstd_threads) --`
    try
        return read(pipeline(IOBuffer(raw), cmd))
    catch
        # Fallback for zstd builds/options that reject -T on this path.
        return read(pipeline(IOBuffer(raw), `zstd -q -c --`))
    end
end

function use_raw_temp_files()::Bool
    raw = get(ENV, "CROCODILE_RAW_TEMP", "0")
    return raw in ("1", "true", "TRUE", "yes", "YES")
end

function compress_index_bucket_files!(index::CanonicalSetIndex)
    for paths in values(index.bucket_by_center)
        for i in eachindex(paths)
            paths[i] = compress_file_zst_inplace(paths[i])
        end
    end
end

function init_canonical_set_index()::CanonicalSetIndex
    return CanonicalSetIndex(
        Dict{NTuple{3, Int16}, Vector{PairWithStart}}(),
        Dict{NTuple{3, Int16}, Vector{String}}(),
        UInt64(0),
        UInt64(0),
    )
end

function append_noncanonical_pair_to_local_buckets!(
    local_bucket_paths::Dict{NTuple{3, Int16}, String},
    tmpdir::String,
    tid::Int,
    pair::PairInfo,
    shared_confrot_mask::Union{Nothing, Vector{UInt64}} = nothing,
)::UInt64
    lp = load_pair(pair)
    n = length(lp.conf)
    n == length(lp.rot) || error("Malformed pair length mismatch: $(pair.pose_path)")
    n == length(lp.offidx) || error("Malformed pair length mismatch: $(pair.pose_path)")

    local_map = Dict{NTuple{3, Int16}, Vector{UInt64}}()
    @inbounds for i in 1:n
        if shared_confrot_mask !== nothing
            bitset_has(shared_confrot_mask::Vector{UInt64}, pack_confrot(lp.conf[i], lp.rot[i])) || continue
        end
        oi = Int(lp.offidx[i]) + 1
        x = lp.abs_offsets[oi, 1]
        y = lp.abs_offsets[oi, 2]
        z = lp.abs_offsets[oi, 3]
        c = canonical_center(x, y, z)
        rx32 = Int32(x) - Int32(c[1])
        ry32 = Int32(y) - Int32(c[2])
        rz32 = Int32(z) - Int32(c[3])
        (rx32 >= -128 && rx32 <= 127 && ry32 >= -128 && ry32 <= 127 && rz32 >= -128 && rz32 <= 127) ||
            error("Canonical rebasing failed for $(pair.pose_path)")
        key = pack_rel_pose_key(lp.conf[i], lp.rot[i], Int8(rx32), Int8(ry32), Int8(rz32))
        recs = get!(local_map, c, UInt64[])
        push!(recs, key)
    end

    temp_bytes = UInt64(0)
    raw_temp = use_raw_temp_files()
    for (center, recs) in local_map
        path = get(local_bucket_paths, center, "")
        if isempty(path)
            suffix = raw_temp ? ".bin" : ".zst"
            path = joinpath(tmpdir, @sprintf("t%02d-%s%s", tid, center_key_filename(center), suffix))
            local_bucket_paths[center] = path
        end
        raw = Vector{UInt8}(undef, length(recs) * 8)
        p = 1
        @inbounds for key in recs
            raw[p] = UInt8(key & 0x00000000000000ff)
            raw[p + 1] = UInt8((key >> 8) & 0x00000000000000ff)
            raw[p + 2] = UInt8((key >> 16) & 0x00000000000000ff)
            raw[p + 3] = UInt8((key >> 24) & 0x00000000000000ff)
            raw[p + 4] = UInt8((key >> 32) & 0x00000000000000ff)
            raw[p + 5] = UInt8((key >> 40) & 0x00000000000000ff)
            raw[p + 6] = UInt8((key >> 48) & 0x00000000000000ff)
            raw[p + 7] = UInt8((key >> 56) & 0x00000000000000ff)
            p += 8
        end
        if raw_temp
            open(path, "a") do io
                write(io, raw)
            end
        else
            zbytes = compress_bytes_zst(raw)
            open(path, "a") do io
                write(io, zbytes)
            end
        end
        temp_bytes += UInt64(length(recs)) * UInt64(8)
    end
    return temp_bytes
end

function build_canonical_index!(
    pairs::Vector{PairInfo},
    tmpdir::String,
    shared_confrot_mask::Union{Nothing, Vector{UInt64}} = nothing,
)::CanonicalSetIndex
    mkpath(tmpdir)
    index = init_canonical_set_index()
    npairs = length(pairs)
    centers = Vector{NTuple{3, Int16}}(undef, npairs)
    total_rows = UInt64(0)
    for i in 1:npairs
        pair = pairs[i]
        center = read_offset_center(pair.offset_path)
        nrows = pose_row_count(pair.pose_path)
        centers[i] = center
        total_rows += UInt64(nrows)
    end

    nt = Threads.maxthreadid()
    local_pairs = [Dict{NTuple{3, Int16}, Vector{PairWithStart}}() for _ in 1:nt]
    local_buckets = [Dict{NTuple{3, Int16}, String}() for _ in 1:nt]
    local_bytes = fill(UInt64(0), nt)

    @threads :static for i in 1:npairs
        tid = threadid()
        center = centers[i]
        pair = pairs[i]
        if is_center_canonical(center)
            vec = get!(local_pairs[tid], center, PairWithStart[])
            push!(vec, PairWithStart(pair))
        else
            local_bytes[tid] += append_noncanonical_pair_to_local_buckets!(
                local_buckets[tid],
                tmpdir,
                tid,
                pair,
                shared_confrot_mask,
            )
        end
    end

    for tid in 1:nt
        for (center, vec) in local_pairs[tid]
            dst = get!(index.pairs_by_center, center, PairWithStart[])
            append!(dst, vec)
        end
        for (center, path) in local_buckets[tid]
            paths = get!(index.bucket_by_center, center, String[])
            push!(paths, path)
        end
        index.temp_bytes += local_bytes[tid]
    end

    index.total_poses = total_rows
    return index
end

function collect_center_keys(index::CanonicalSetIndex)::Set{NTuple{3, Int16}}
    out = Set{NTuple{3, Int16}}()
    for c in keys(index.pairs_by_center)
        push!(out, c)
    end
    for c in keys(index.bucket_by_center)
        push!(out, c)
    end
    return out
end

function load_center_keys(
    index::CanonicalSetIndex,
    center::NTuple{3, Int16},
)::Vector{UInt64}
    out = UInt64[]
    pair_refs = get(index.pairs_by_center, center, PairWithStart[])
    for pref in pair_refs
        lp = load_pair(pref.pair)
        lp.center == center || error("Canonical pair center mismatch for $(pref.pair.pose_path)")
        n = length(lp.conf)
        sizehint!(out, length(out) + n)
        @inbounds for i in 1:n
            oi = Int(lp.offidx[i]) + 1
            key = pack_rel_pose_key(
                lp.conf[i],
                lp.rot[i],
                lp.rel_offsets[oi, 1],
                lp.rel_offsets[oi, 2],
                lp.rel_offsets[oi, 3],
            )
            push!(out, key)
        end
    end
    paths = get(index.bucket_by_center, center, String[])
    for path in paths
        append!(out, read_key_records(path))
    end
    return out
end

function load_center_keys_filtered(
    index::CanonicalSetIndex,
    center::NTuple{3, Int16},
    shared_confrot_mask::Vector{UInt64},
)::Vector{UInt64}
    out = UInt64[]
    pair_refs = get(index.pairs_by_center, center, PairWithStart[])
    for pref in pair_refs
        lp = load_pair(pref.pair)
        lp.center == center || error("Canonical pair center mismatch for $(pref.pair.pose_path)")
        n = length(lp.conf)
        @inbounds for i in 1:n
            conf = lp.conf[i]
            rot = lp.rot[i]
            bitset_has(shared_confrot_mask, pack_confrot(conf, rot)) || continue
            oi = Int(lp.offidx[i]) + 1
            key = pack_rel_pose_key(
                conf,
                rot,
                lp.rel_offsets[oi, 1],
                lp.rel_offsets[oi, 2],
                lp.rel_offsets[oi, 3],
            )
            push!(out, key)
        end
    end
    paths = get(index.bucket_by_center, center, String[])
    for path in paths
        keys = read_key_records(path)
        @inbounds for k in keys
            bitset_has(shared_confrot_mask, key_confrot(k)) || continue
            push!(out, k)
        end
    end
    return out
end

function center_inner_pass1_min_rows()::Int
    try
        return max(1, parse(Int, get(ENV, "CROCODILE_CENTER_INNER_PASS1_MIN_ROWS", "2000000")))
    catch
        return 2_000_000
    end
end

function enable_global_confrot_prefilter()::Bool
    raw = get(ENV, "CROCODILE_GLOBAL_CONFROT_PREFILTER", "0")
    return raw in ("1", "true", "TRUE", "yes", "YES")
end

function enable_center_inner_pass1()::Bool
    raw = get(ENV, "CROCODILE_CENTER_INNER_PASS1", "0")
    return raw in ("1", "true", "TRUE", "yes", "YES")
end

function enable_map_confrot_prefilter()::Bool
    raw = get(ENV, "CROCODILE_MAP_CONFROT_PREFILTER", "1")
    return raw in ("1", "true", "TRUE", "yes", "YES")
end

@inline function pack_confrot(conf::UInt16, rot::UInt16)::UInt32
    return UInt32(conf) | (UInt32(rot) << 16)
end

@inline function key_confrot(key::UInt64)::UInt32
    return UInt32(key & 0x0000_0000_ffff_ffff)
end

@inline function bitset_word_and_bit(v::UInt32)::Tuple{Int, UInt64}
    vv = UInt64(v)
    wi = Int((vv >>> 6) + 1)
    bit = UInt64(1) << Int(vv & 0x3f)
    return wi, bit
end

@inline function bitset_has(mask::Vector{UInt64}, v::UInt32)::Bool
    wi, bit = bitset_word_and_bit(v)
    return (mask[wi] & bit) != 0
end

@inline function bitset_set_new!(mask::Vector{UInt64}, v::UInt32)::Bool
    wi, bit = bitset_word_and_bit(v)
    old = mask[wi]
    new = old | bit
    mask[wi] = new
    return new != old
end

function confrot_mask_words()::Int
    # 2^32 values, one bit each
    return Int((UInt64(typemax(UInt32)) + UInt64(64)) >>> 6)
end

function update_confrot_mask_from_index!(
    mask::Vector{UInt64},
    index::CanonicalSetIndex,
)::UInt64
    uniq = UInt64(0)
    for pair_refs in values(index.pairs_by_center)
        for pref in pair_refs
            conf, rot, _ = read_pose_columns(pref.pair.pose_path)
            @inbounds for i in eachindex(conf)
                uniq += bitset_set_new!(mask, pack_confrot(conf[i], rot[i])) ? UInt64(1) : UInt64(0)
            end
        end
    end
    for paths in values(index.bucket_by_center)
        for path in paths
            keys = read_key_records(path)
            @inbounds for k in keys
                uniq += bitset_set_new!(mask, key_confrot(k)) ? UInt64(1) : UInt64(0)
            end
        end
    end
    return uniq
end

function update_confrot_mask_from_pairs!(
    mask::Vector{UInt64},
    pairs::Vector{PairInfo},
)::UInt64
    uniq = UInt64(0)
    for pair in pairs
        conf, rot, _ = read_pose_columns(pair.pose_path)
        @inbounds for i in eachindex(conf)
            uniq += bitset_set_new!(mask, pack_confrot(conf[i], rot[i])) ? UInt64(1) : UInt64(0)
        end
    end
    return uniq
end

function build_global_confrot_shared_mask_from_pairs(
    pairs_a::Vector{PairInfo},
    pairs_b::Vector{PairInfo},
)::Tuple{Vector{UInt64}, UInt64, UInt64}
    nwords = confrot_mask_words()
    mask_a = zeros(UInt64, nwords)
    shared = zeros(UInt64, nwords)
    uniq_a = update_confrot_mask_from_pairs!(mask_a, pairs_a)
    shared_count = UInt64(0)
    for pair in pairs_b
        conf, rot, _ = read_pose_columns(pair.pose_path)
        @inbounds for i in eachindex(conf)
            cr = pack_confrot(conf[i], rot[i])
            if bitset_has(mask_a, cr)
                shared_count += bitset_set_new!(shared, cr) ? UInt64(1) : UInt64(0)
            end
        end
    end
    return shared, uniq_a, shared_count
end

function build_global_confrot_shared_mask(
    index_a::CanonicalSetIndex,
    index_b::CanonicalSetIndex,
)::Tuple{Vector{UInt64}, UInt64, UInt64}
    nwords = confrot_mask_words()
    mask_a = zeros(UInt64, nwords)
    shared = zeros(UInt64, nwords)
    uniq_a = update_confrot_mask_from_index!(mask_a, index_a)

    shared_count = UInt64(0)
    for pair_refs in values(index_b.pairs_by_center)
        for pref in pair_refs
            conf, rot, _ = read_pose_columns(pref.pair.pose_path)
            @inbounds for i in eachindex(conf)
                cr = pack_confrot(conf[i], rot[i])
                if bitset_has(mask_a, cr)
                    shared_count += bitset_set_new!(shared, cr) ? UInt64(1) : UInt64(0)
                end
            end
        end
    end
    for paths in values(index_b.bucket_by_center)
        for path in paths
            keys = read_key_records(path)
            @inbounds for k in keys
                cr = key_confrot(k)
                if bitset_has(mask_a, cr)
                    shared_count += bitset_set_new!(shared, cr) ? UInt64(1) : UInt64(0)
                end
            end
        end
    end
    return shared, uniq_a, shared_count
end

function update_confrot_mask_from_lookup_file!(
    mask::Vector{UInt64},
    path::String,
)::UInt64
    isfile(path) || return UInt64(0)
    bytes = if endswith(path, ".zst")
        decompress_zst_bytes(path; single_thread = true)
    else
        read(path)
    end
    sz = length(bytes)
    sz == 0 && return UInt64(0)
    sz % 12 == 0 || error("Corrupt keyuid file size: $path")

    uniq = UInt64(0)
    p = 1
    @inbounds while p <= sz
        cr = read_le_u32(bytes, p)
        uniq += bitset_set_new!(mask, cr) ? UInt64(1) : UInt64(0)
        p += 12
    end
    return uniq
end

function build_map_confrot_mask_from_lookup_paths(
    lookup_paths::Dict{NTuple{3, Int16}, String},
)::Tuple{Vector{UInt64}, UInt64}
    mask = zeros(UInt64, confrot_mask_words())
    uniq = UInt64(0)
    for path in values(lookup_paths)
        uniq += update_confrot_mask_from_lookup_file!(mask, path)
    end
    return mask, uniq
end

function filter_keys_by_confrot!(
    keys::Vector{UInt64},
    shared_mask::Vector{UInt64},
)::Int
    n = length(keys)
    w = 1
    @inbounds for r in 1:n
        k = keys[r]
        if bitset_has(shared_mask, key_confrot(k))
            keys[w] = k
            w += 1
        end
    end
    resize!(keys, w - 1)
    return w - 1
end

function build_shared_confrot_set_from_keys(
    keys_a::Vector{UInt64},
    keys_b::Vector{UInt64},
)::Tuple{Set{UInt32}, UInt64, UInt64}
    seen_a = Set{UInt32}()
    sizehint!(seen_a, min(length(keys_a), 2_000_000))
    @inbounds for k in keys_a
        push!(seen_a, key_confrot(k))
    end

    shared = Set{UInt32}()
    sizehint!(shared, min(length(keys_b), length(seen_a)))
    @inbounds for k in keys_b
        cr = key_confrot(k)
        if cr in seen_a
            push!(shared, cr)
        end
    end
    return shared, UInt64(length(seen_a)), UInt64(length(shared))
end

function filter_keys_by_confrot_set!(
    keys::Vector{UInt64},
    shared_set::Set{UInt32},
)::Int
    n = length(keys)
    w = 1
    @inbounds for r in 1:n
        k = keys[r]
        if key_confrot(k) in shared_set
            keys[w] = k
            w += 1
        end
    end
    resize!(keys, w - 1)
    return w - 1
end

function center_inner_pass2_min_rows()::Int
    try
        return max(1, parse(Int, get(ENV, "CROCODILE_CENTER_INNER_PASS2_MIN_ROWS", "2000000")))
    catch
        return 2_000_000
    end
end

function center_inner_pass2_chunk_rows()::Int
    try
        return max(1, parse(Int, get(ENV, "CROCODILE_CENTER_INNER_PASS2_CHUNK_ROWS", "1000000")))
    catch
        return 1_000_000
    end
end

function enable_center_inner_pass2()::Bool
    raw = get(ENV, "CROCODILE_CENTER_INNER_PASS2", "0")
    return raw in ("1", "true", "TRUE", "yes", "YES")
end

function intersect_sorted_unique_serial(
    keys_a::Vector{UInt64},
    keys_b::Vector{UInt64},
)::Vector{UInt64}
    out = UInt64[]
    sizehint!(out, min(length(keys_a), length(keys_b)))
    ia = 1
    ib = 1
    na = length(keys_a)
    nb = length(keys_b)
    while ia <= na && ib <= nb
        ka = keys_a[ia]
        kb = keys_b[ib]
        if ka < kb
            ia += 1
        elseif kb < ka
            ib += 1
        else
            push!(out, ka)
            ia += 1
            ib += 1
        end
    end
    return out
end

function intersect_sorted_unique_parallel(
    keys_a::Vector{UInt64},
    keys_b::Vector{UInt64},
)::Vector{UInt64}
    na = length(keys_a)
    nb = length(keys_b)
    if Threads.nthreads() <= 1 || min(na, nb) < center_inner_pass1_min_rows()
        return intersect_sorted_unique_serial(keys_a, keys_b)
    end

    nparts = min(Threads.nthreads(), 8)
    nparts <= 1 && return intersect_sorted_unique_serial(keys_a, keys_b)
    big = na >= nb ? keys_a : keys_b

    pivots = Vector{UInt64}(undef, nparts - 1)
    @inbounds for p in 1:(nparts - 1)
        idx = cld(p * length(big), nparts)
        pivots[p] = big[idx]
    end

    a_starts = Vector{Int}(undef, nparts + 1)
    b_starts = Vector{Int}(undef, nparts + 1)
    a_starts[1] = 1
    b_starts[1] = 1
    @inbounds for p in 1:(nparts - 1)
        a_starts[p + 1] = searchsortedfirst(keys_a, pivots[p])
        b_starts[p + 1] = searchsortedfirst(keys_b, pivots[p])
    end
    a_starts[nparts + 1] = na + 1
    b_starts[nparts + 1] = nb + 1

    locals = [UInt64[] for _ in 1:nparts]
    @sync for part in 1:nparts
        Threads.@spawn begin
            alo = a_starts[part]
            ahi = a_starts[part + 1] - 1
            blo = b_starts[part]
            bhi = b_starts[part + 1] - 1
            if alo > ahi || blo > bhi
                locals[part] = UInt64[]
                return
            end

            out = UInt64[]
            sizehint!(out, min(ahi - alo + 1, bhi - blo + 1))
            ia = alo
            ib = blo
            while ia <= ahi && ib <= bhi
                ka = keys_a[ia]
                kb = keys_b[ib]
                if ka < kb
                    ia += 1
                elseif kb < ka
                    ib += 1
                else
                    push!(out, ka)
                    ia += 1
                    ib += 1
                end
            end
            locals[part] = out
        end
    end

    total = 0
    for v in locals
        total += length(v)
    end
    out = Vector{UInt64}(undef, total)
    pos = 1
    for v in locals
        n = length(v)
        n == 0 && continue
        copyto!(out, pos, v, 1, n)
        pos += n
    end
    return out
end

function intersect_center_keys(
    index_a::CanonicalSetIndex,
    index_b::CanonicalSetIndex,
    center::NTuple{3, Int16};
    allow_inner_parallel::Bool = false,
    shared_confrot_mask::Union{Nothing, Vector{UInt64}} = nothing,
    local_confrot_prefilter::Bool = false,
)::Vector{UInt64}
    keys_a = shared_confrot_mask === nothing ?
             load_center_keys(index_a, center) :
             load_center_keys_filtered(index_a, center, shared_confrot_mask::Vector{UInt64})
    keys_b = shared_confrot_mask === nothing ?
             load_center_keys(index_b, center) :
             load_center_keys_filtered(index_b, center, shared_confrot_mask::Vector{UInt64})
    isempty(keys_a) && return UInt64[]
    isempty(keys_b) && return UInt64[]
    allow_inner_parallel && println("stage:mark unique-k-pass1-center-load-done")
    if shared_confrot_mask === nothing && local_confrot_prefilter
        shared_set, uniq_confrot_a, shared_confrot = build_shared_confrot_set_from_keys(keys_a, keys_b)
        filter_keys_by_confrot_set!(keys_a, shared_set)
        filter_keys_by_confrot_set!(keys_b, shared_set)
        allow_inner_parallel && println("center confrot prefilter: unique-A=$uniq_confrot_a shared=$shared_confrot")
        isempty(keys_a) && return UInt64[]
        isempty(keys_b) && return UInt64[]
    end

    use_inner = allow_inner_parallel &&
                enable_center_inner_pass1() &&
                Threads.nthreads() > 1 &&
                (length(keys_a) + length(keys_b)) >= center_inner_pass1_min_rows()
    if use_inner
        println("stage:mark unique-k-pass1-center-sort-start")
        t = Threads.@spawn sort!(keys_a)
        sort!(keys_b)
        fetch(t)
        t2 = Threads.@spawn unique!(keys_a)
        unique!(keys_b)
        fetch(t2)
        isempty(keys_a) && return UInt64[]
        isempty(keys_b) && return UInt64[]
        println("stage:mark unique-k-pass1-center-sortuniq-done")
        println("stage:mark unique-k-pass1-center-intersect-start")
        out = intersect_sorted_unique_serial(keys_a, keys_b)
        println("stage:mark unique-k-pass1-center-intersect-done")
        return out
    else
        sort!(keys_a)
        sort!(keys_b)
        unique!(keys_a)
        unique!(keys_b)
        isempty(keys_a) && return UInt64[]
        isempty(keys_b) && return UInt64[]
        allow_inner_parallel && println("stage:mark unique-k-pass1-center-sortuniq-done")
        allow_inner_parallel && println("stage:mark unique-k-pass1-center-intersect-start")
        out = intersect_sorted_unique_serial(keys_a, keys_b)
        allow_inner_parallel && println("stage:mark unique-k-pass1-center-intersect-done")
        return out
    end
end

function append_file_to_io!(out::IO, path::String, buf::Vector{UInt8})
    open(path, "r") do io
        while true
            n = readbytes!(io, buf)
            n == 0 && break
            write(out, view(buf, 1:n))
        end
    end
end

function write_center_lookup_and_k_range!(
    keys::Vector{UInt64},
    j1::Int,
    j2::Int,
    center::NTuple{3, Int16},
    base::UInt64,
    part_dir::String,
    lpath::String,
    cfg::CliConfig,
    raw_temp::Bool,
)
    j1 > j2 && return
    k_part_writer = init_pose_writer(
        part_dir;
        start_index = 1,
        max_poses_per_chunk = cfg.max_poses_per_chunk,
        write_empty_if_none = false,
    )
    nrows = j2 - j1 + 1
    lookup_bytes = Vector{UInt8}(undef, 12 * nrows)
    p = 1
    @inbounds for j in j1:j2
        uid64 = base + UInt64(j - 1)
        uid64 <= UInt64(typemax(UInt32)) || error("Unique-K index exceeds UInt32 range")
        uid = UInt32(uid64)
        ka = keys[j]
        conf, rot, rx, ry, rz = unpack_rel_pose_key(ka)
        x32 = Int32(center[1]) + Int32(rx)
        y32 = Int32(center[2]) + Int32(ry)
        z32 = Int32(center[3]) + Int32(rz)
        (x32 >= typemin(Int16) && x32 <= typemax(Int16)) || error("x out of int16 range")
        (y32 >= typemin(Int16) && y32 <= typemax(Int16)) || error("y out of int16 range")
        (z32 >= typemin(Int16) && z32 <= typemax(Int16)) || error("z out of int16 range")
        add_pose!(k_part_writer, conf, rot, Int16(x32), Int16(y32), Int16(z32))
        write_keyuid_to_bytes!(lookup_bytes, p, ka, uid)
        p += 12
    end
    if raw_temp
        open(lpath, "w") do lio
            write(lio, lookup_bytes)
        end
    else
        open_zstd_writer(lpath; single_thread = zstd_should_use_single_thread(length(lookup_bytes))) do lio
            write(lio, lookup_bytes)
        end
    end
    finish_pose_writer!(k_part_writer)
end

function write_center_lookup_and_k_parallel!(
    keys::Vector{UInt64},
    center::NTuple{3, Int16},
    base::UInt64,
    part_dir::String,
    lpath::String,
    cfg::CliConfig,
    raw_temp::Bool,
)
    n = length(keys)
    chunk_rows = center_inner_pass2_chunk_rows()
    nchunks = min(Threads.nthreads(), max(2, cld(n, chunk_rows)))
    sub_dirs = fill("", nchunks)
    sub_lookup_paths = fill("", nchunks)

    @sync for c in 1:nchunks
        Threads.@spawn begin
            j1 = (c - 1) * chunk_rows + 1
            j2 = min(n, c * chunk_rows)
            j1 > j2 && return

            sub_dir = joinpath(part_dir, @sprintf("chunk-%03d", c))
            mkpath(sub_dir)
            sub_lookup = lpath * @sprintf(".part%03d", c)
            write_center_lookup_and_k_range!(keys, j1, j2, center, base, sub_dir, sub_lookup, cfg, raw_temp)
            sub_dirs[c] = sub_dir
            sub_lookup_paths[c] = sub_lookup
        end
    end

    buf = Vector{UInt8}(undef, 1 << 20)
    open(lpath, "w") do out
        for c in 1:nchunks
            p = sub_lookup_paths[c]
            isempty(p) && continue
            append_file_to_io!(out, p, buf)
            rm(p; force = true)
        end
    end

    next_idx = 1
    for c in 1:nchunks
        sub_dir = sub_dirs[c]
        isempty(sub_dir) && continue
        pairs = discover_pose_pairs(sub_dir)
        for pair in pairs
            pose_dst = joinpath(part_dir, "poses-$(next_idx).npy.zst")
            offset_dst = joinpath(part_dir, "offsets-$(next_idx).dat")
            mv(pair.pose_path, pose_dst; force = true)
            mv(pair.offset_path, offset_dst; force = true)
            next_idx += 1
        end
        rm(sub_dir; recursive = true, force = true)
    end
end

function build_unique_k_and_lookup!(
    index_a::CanonicalSetIndex,
    index_b::CanonicalSetIndex,
    outdir::String,
    lookup_dir::String,
    cfg::CliConfig,
    shared_confrot_mask_in::Union{Nothing, Vector{UInt64}} = nothing,
)::Tuple{UInt64, Dict{NTuple{3, Int16}, String}}
    mkpath(lookup_dir)
    raw_temp = use_raw_temp_files()
    inter_ext = raw_temp ? "" : ".zst"
    lookup_ext = raw_temp ? "" : ".zst"

    centers_a = collect_center_keys(index_a)
    centers_b = collect_center_keys(index_b)
    common = sort!(collect(intersect(centers_a, centers_b)))
    ncommon = length(common)
    confrot_shared_mask::Union{Nothing, Vector{UInt64}} = shared_confrot_mask_in
    local_confrot_prefilter = false
    if confrot_shared_mask !== nothing
        println("stage:mark unique-k-confrot-prefilter-reuse")
    elseif enable_global_confrot_prefilter()
        force_global = let raw = get(ENV, "CROCODILE_CONFROT_FORCE_GLOBAL", "0")
            raw in ("1", "true", "TRUE", "yes", "YES")
        end
        min_rows_global = try
            max(1, parse(Int, get(ENV, "CROCODILE_CONFROT_GLOBAL_MIN_ROWS", "1500000000")))
        catch
            1_500_000_000
        end
        total_rows_est = index_a.total_poses + index_b.total_poses
        use_global = force_global || (total_rows_est >= UInt64(min_rows_global))
        if use_global
            println("stage:mark unique-k-confrot-prefilter-start")
            confrot_shared_mask, uniq_confrot_a, shared_confrot = build_global_confrot_shared_mask(index_a, index_b)
            println("unique-k confrot prefilter: unique-A=$uniq_confrot_a shared=$shared_confrot")
            println("stage:mark unique-k-confrot-prefilter-end")
        else
            local_confrot_prefilter = true
            println("stage:mark unique-k-confrot-prefilter-center-local")
        end
    end
    lookup_paths = Dict{NTuple{3, Int16}, String}()
    inter_dir = joinpath(lookup_dir, "_intersections")
    mkpath(inter_dir)

    inter_paths = fill("", ncommon)
    inter_counts = fill(UInt64(0), ncommon)
    progress_pass1 = Threads.Atomic{Int}(0)
    q25 = ncommon > 0 ? cld(ncommon, 4) : 0
    q50 = ncommon > 0 ? cld(ncommon, 2) : 0
    q75 = ncommon > 0 ? cld(3 * ncommon, 4) : 0
    if ncommon > 0
        ch1 = Channel{Int}(ncommon)
        for i in 1:ncommon
            put!(ch1, i)
        end
        close(ch1)
        nworkers1 = min(nthreads(), ncommon)
        @sync for _ in 1:nworkers1
            Threads.@spawn begin
                for i in ch1
                    center = common[i]
                    keys = intersect_center_keys(
                        index_a,
                        index_b,
                        center;
                        allow_inner_parallel = (ncommon == 1),
                        shared_confrot_mask = confrot_shared_mask,
                        local_confrot_prefilter = local_confrot_prefilter,
                    )
                    if !isempty(keys)
                        p = joinpath(inter_dir, center_key_filename(center) * inter_ext)
                        if raw_temp
                            write_u64_records_raw(p, keys)
                        else
                            write_u64_records_zst(
                                p,
                                keys;
                                single_thread = zstd_should_use_single_thread(length(keys) * 8),
                            )
                        end
                        inter_paths[i] = p
                        inter_counts[i] = UInt64(length(keys))
                    end
                    done = Threads.atomic_add!(progress_pass1, 1) + 1
                    done == q25 && println("stage:mark unique-k-pass1-25")
                    done == q50 && println("stage:mark unique-k-pass1-50")
                    done == q75 && println("stage:mark unique-k-pass1-75")
                    done == ncommon && println("stage:mark unique-k-pass1-100")
                end
            end
        end
    end
    confrot_shared_mask = nothing

    uid_starts = fill(UInt64(0), ncommon)
    uid_next = UInt64(0)
    for i in 1:ncommon
        uid_starts[i] = uid_next
        uid_next += inter_counts[i]
    end
    uid_next <= UInt64(typemax(UInt32)) + UInt64(1) || error("Unique-K index exceeds UInt32 range")
    println("stage:mark unique-k-pass2-start")

    k_parts_dir = joinpath(lookup_dir, "_kparts")
    mkpath(k_parts_dir)
    part_dirs = fill("", ncommon)
    part_lookup_paths = fill("", ncommon)
    progress_pass2 = Threads.Atomic{Int}(0)
    q25_2 = ncommon > 0 ? cld(ncommon, 4) : 0
    q50_2 = ncommon > 0 ? cld(ncommon, 2) : 0
    q75_2 = ncommon > 0 ? cld(3 * ncommon, 4) : 0

    if ncommon > 0
        ch2 = Channel{Int}(ncommon)
        for i in 1:ncommon
            put!(ch2, i)
        end
        close(ch2)
        nworkers2 = min(nthreads(), ncommon)
        if enable_center_inner_pass2() && nworkers2 > 1
            nworkers2 = min(nworkers2, max(1, nthreads() ÷ 2))
        end
        avg_per_thread = nthreads() > 0 ? cld(uid_next, UInt64(nthreads())) : uid_next
        @sync for _ in 1:nworkers2
            Threads.@spawn begin
                for i in ch2
                    if inter_counts[i] != 0
                        center = common[i]
                        keys = read_key_records(
                            inter_paths[i];
                            single_thread = zstd_should_use_single_thread(Int(inter_counts[i] * UInt64(8))),
                        )
                        UInt64(length(keys)) == inter_counts[i] || error("Intersection key count mismatch for center $(center)")

                        part_dir = joinpath(k_parts_dir, center_key_filename(center))
                        mkpath(part_dir)
                        lpath = joinpath(lookup_dir, center_key_filename(center) * lookup_ext)
                        base = uid_starts[i]
                        inner_pass2 = enable_center_inner_pass2() &&
                                      Threads.nthreads() > 1 &&
                                      length(keys) >= center_inner_pass2_min_rows() &&
                                      inter_counts[i] >= avg_per_thread
                        if inner_pass2
                            write_center_lookup_and_k_parallel!(keys, center, base, part_dir, lpath, cfg, raw_temp)
                        else
                            write_center_lookup_and_k_range!(
                                keys,
                                1,
                                length(keys),
                                center,
                                base,
                                part_dir,
                                lpath,
                                cfg,
                                raw_temp,
                            )
                        end
                        part_dirs[i] = part_dir
                        part_lookup_paths[i] = lpath
                        rm(inter_paths[i]; force = true)
                    end
                    done = Threads.atomic_add!(progress_pass2, 1) + 1
                    done == q25_2 && println("stage:mark unique-k-pass2-25")
                    done == q50_2 && println("stage:mark unique-k-pass2-50")
                    done == q75_2 && println("stage:mark unique-k-pass2-75")
                    done == ncommon && println("stage:mark unique-k-pass2-100")
                end
            end
        end
    end
    rm(inter_dir; recursive = true, force = true)
    println("stage:mark unique-k-merge-start")

    next_index = 1
    for i in 1:ncommon
        part_dirs[i] == "" && continue
        pairs = discover_pose_pairs(part_dirs[i])
        for pair in pairs
            pose_dst = joinpath(outdir, "poses-$(next_index).npy.zst")
            offset_dst = joinpath(outdir, "offsets-$(next_index).dat")
            mv(pair.pose_path, pose_dst; force = true)
            mv(pair.offset_path, offset_dst; force = true)
            next_index += 1
        end
        rm(part_dirs[i]; recursive = true, force = true)
        lookup_paths[common[i]] = part_lookup_paths[i]
    end
    rm(k_parts_dir; recursive = true, force = true)

    if next_index == 1
        k_writer = init_pose_writer(
            outdir;
            start_index = 1,
            max_poses_per_chunk = cfg.max_poses_per_chunk,
            write_empty_if_none = true,
        )
        finish_pose_writer!(k_writer)
    end
    return uid_next, lookup_paths
end

function get_lookup_for_center!(
    cache::Dict{NTuple{3, Int16}, Dict{UInt64, UInt32}},
    order::Vector{NTuple{3, Int16}},
    lookup_paths::Dict{NTuple{3, Int16}, String},
    center::NTuple{3, Int16},
    max_cached::Int,
)::Union{Nothing, Dict{UInt64, UInt32}}
    existing = get(cache, center, nothing)
    existing !== nothing && return existing::Dict{UInt64, UInt32}

    path = get(lookup_paths, center, "")
    isempty(path) && return nothing
    fsz = try
        Int(stat(path).size)
    catch
        0
    end
    keys, uids = read_keyuid_records(path; single_thread = zstd_should_use_single_thread(fsz))
    d = Dict{UInt64, UInt32}()
    sizehint!(d, length(keys))
    @inbounds for i in eachindex(keys)
        d[keys[i]] = uids[i]
    end
    cache[center] = d
    push!(order, center)
    if max_cached > 0 && length(order) > max_cached
        old = popfirst!(order)
        old != center && delete!(cache, old)
    end
    return d
end

function get_lookup_for_center_local!(
    cache::Dict{NTuple{3, Int16}, Dict{UInt64, UInt32}},
    missing::Set{NTuple{3, Int16}},
    lookup_paths::Dict{NTuple{3, Int16}, String},
    center::NTuple{3, Int16},
)::Union{Nothing, Dict{UInt64, UInt32}}
    (center in missing) && return nothing
    existing = get(cache, center, nothing)
    existing !== nothing && return existing::Dict{UInt64, UInt32}

    path = get(lookup_paths, center, "")
    if isempty(path)
        push!(missing, center)
        return nothing
    end
    keys, uids = read_keyuid_records(path; single_thread = true)
    d = Dict{UInt64, UInt32}()
    sizehint!(d, length(keys))
    @inbounds for i in eachindex(keys)
        d[keys[i]] = uids[i]
    end
    cache[center] = d
    return d
end

function build_noncanonical_offset_lookup_info(
    lp::LoadedPair,
    lookup_paths::Dict{NTuple{3, Int16}, String},
    cache::Dict{NTuple{3, Int16}, Dict{UInt64, UInt32}},
    order::Vector{NTuple{3, Int16}},
    max_cached::Int,
)::Tuple{Vector{UInt64}, Vector{Int32}, Vector{Dict{UInt64, UInt32}}}
    noff = size(lp.abs_offsets, 1)
    off_suffix = Vector{UInt64}(undef, noff)
    off_lookup_id = Vector{Int32}(undef, noff)

    center_to_id = Dict{NTuple{3, Int16}, Int32}()
    lookups = Dict{UInt64, UInt32}[]

    @inbounds for oi in 1:noff
        x = lp.abs_offsets[oi, 1]
        y = lp.abs_offsets[oi, 2]
        z = lp.abs_offsets[oi, 3]
        center = canonical_center(x, y, z)
        rx32 = Int32(x) - Int32(center[1])
        ry32 = Int32(y) - Int32(center[2])
        rz32 = Int32(z) - Int32(center[3])
        (rx32 >= -128 && rx32 <= 127 && ry32 >= -128 && ry32 <= 127 && rz32 >= -128 && rz32 <= 127) ||
            error("Canonical rebasing failed for offsets in center $(lp.center)")
        off_suffix[oi] = pack_rel_suffix(Int8(rx32), Int8(ry32), Int8(rz32))

        id = get(center_to_id, center, Int32(-1))
        if id == Int32(-1)
            lookup = get_lookup_for_center!(cache, order, lookup_paths, center, max_cached)
            if lookup === nothing
                id = Int32(0)
            else
                push!(lookups, lookup::Dict{UInt64, UInt32})
                id = Int32(length(lookups))
            end
            center_to_id[center] = id
        end
        off_lookup_id[oi] = id
    end
    return off_suffix, off_lookup_id, lookups
end

function map_set_to_lookup!(
    pairs::Vector{PairInfo},
    lookup_paths::Dict{NTuple{3, Int16}, String},
    outdir::String,
    prefix::String,
    cfg::CliConfig,
    ;
    total_poses::Union{Nothing, UInt64} = nothing,
    confrot_mask::Union{Nothing, Vector{UInt64}} = nothing,
)::UInt64
    use_u32 = total_poses !== nothing && (total_poses::UInt64) <= (UInt64(typemax(UInt32)) + UInt64(1))
    writer64 = use_u32 ? nothing : init_index64_writer(outdir, prefix; start_index = 1, flush_rows = cfg.threshold_k)
    writer32 = use_u32 ? init_index_writer(outdir, prefix; start_index = 1, ncols = 2, flush_rows = cfg.threshold_k) : nothing
    max_cached = try
        parse(Int, get(ENV, "CROCODILE_LOOKUP_CACHE_CENTERS", "64"))
    catch
        64
    end
    max_cached = max(0, max_cached)
    min_rows_threaded = try
        parse(Int, get(ENV, "CROCODILE_MAP_THREADS_MIN_ROWS", "500000"))
    catch
        500_000
    end
    min_rows_threaded = max(1, min_rows_threaded)
    cache = Dict{NTuple{3, Int16}, Dict{UInt64, UInt32}}()
    order = NTuple{3, Int16}[]
    use_confrot_mask = confrot_mask !== nothing
    mask = confrot_mask

    total_hits = UInt64(0)
    global_start = UInt64(0)
    for pair in pairs
        lp = load_pair(pair)
        n = length(lp.conf)
        out_idx = UInt64[]
        out_uid = UInt64[]
        sizehint!(out_idx, n ÷ 16 + 1)
        sizehint!(out_uid, n ÷ 16 + 1)

        pair_center = lp.center
        pair_is_canonical = is_center_canonical(pair_center)
        if pair_is_canonical
            lookup = get_lookup_for_center!(cache, order, lookup_paths, pair_center, max_cached)
            if lookup !== nothing
                d = lookup::Dict{UInt64, UInt32}
                noff = size(lp.rel_offsets, 1)
                rel_suffix = Vector{UInt64}(undef, noff)
                @inbounds for oi in 1:noff
                    rel_suffix[oi] = pack_rel_suffix(lp.rel_offsets[oi, 1], lp.rel_offsets[oi, 2], lp.rel_offsets[oi, 3])
                end
                if Threads.nthreads() > 1 && n >= min_rows_threaded
                    nt = Threads.maxthreadid()
                    local_idx = [UInt64[] for _ in 1:nt]
                    local_uid = [UInt64[] for _ in 1:nt]
                    hint = max(16, n ÷ max(1, nt * 8))
                    @inbounds for tid in 1:nt
                        sizehint!(local_idx[tid], hint)
                        sizehint!(local_uid[tid], hint)
                    end
                    if use_confrot_mask
                        @threads :static for i in 1:n
                            tid = threadid()
                            oi = Int(lp.offidx[i]) + 1
                            cr = pack_confrot(lp.conf[i], lp.rot[i])
                            bitset_has(mask::Vector{UInt64}, cr) || continue
                            key = UInt64(cr) | rel_suffix[oi]
                            uid = get(d, key, UInt32(0xffffffff))
                            uid == UInt32(0xffffffff) && continue
                            push!(local_idx[tid], global_start + UInt64(i - 1))
                            push!(local_uid[tid], UInt64(uid))
                        end
                    else
                        @threads :static for i in 1:n
                            tid = threadid()
                            oi = Int(lp.offidx[i]) + 1
                            cr = pack_confrot(lp.conf[i], lp.rot[i])
                            key = UInt64(cr) | rel_suffix[oi]
                            uid = get(d, key, UInt32(0xffffffff))
                            uid == UInt32(0xffffffff) && continue
                            push!(local_idx[tid], global_start + UInt64(i - 1))
                            push!(local_uid[tid], UInt64(uid))
                        end
                    end
                    @inbounds for tid in 1:nt
                        append!(out_idx, local_idx[tid])
                        append!(out_uid, local_uid[tid])
                    end
                else
                    if use_confrot_mask
                        @inbounds for i in 1:n
                            oi = Int(lp.offidx[i]) + 1
                            cr = pack_confrot(lp.conf[i], lp.rot[i])
                            bitset_has(mask::Vector{UInt64}, cr) || continue
                            key = UInt64(cr) | rel_suffix[oi]
                            uid = get(d, key, UInt32(0xffffffff))
                            uid == UInt32(0xffffffff) && continue
                            push!(out_idx, global_start + UInt64(i - 1))
                            push!(out_uid, UInt64(uid))
                        end
                    else
                        @inbounds for i in 1:n
                            oi = Int(lp.offidx[i]) + 1
                            cr = pack_confrot(lp.conf[i], lp.rot[i])
                            key = UInt64(cr) | rel_suffix[oi]
                            uid = get(d, key, UInt32(0xffffffff))
                            uid == UInt32(0xffffffff) && continue
                            push!(out_idx, global_start + UInt64(i - 1))
                            push!(out_uid, UInt64(uid))
                        end
                    end
                end
            end
        else
            off_suffix, off_lookup_id, off_lookups = build_noncanonical_offset_lookup_info(
                lp,
                lookup_paths,
                cache,
                order,
                max_cached,
            )
            isempty(off_lookups) || begin
            if Threads.nthreads() > 1 && n >= min_rows_threaded
                nt = Threads.maxthreadid()
                local_idx = [UInt64[] for _ in 1:nt]
                local_uid = [UInt64[] for _ in 1:nt]
                hint = max(16, n ÷ max(1, nt * 8))
                @inbounds for tid in 1:nt
                    sizehint!(local_idx[tid], hint)
                    sizehint!(local_uid[tid], hint)
                end
                if use_confrot_mask
                    @threads :static for i in 1:n
                        tid = threadid()
                        oi = Int(lp.offidx[i]) + 1
                        lid = off_lookup_id[oi]
                        lid == 0 && continue
                        cr = pack_confrot(lp.conf[i], lp.rot[i])
                        bitset_has(mask::Vector{UInt64}, cr) || continue
                        key = UInt64(cr) | off_suffix[oi]
                        uid = get(off_lookups[Int(lid)], key, UInt32(0xffffffff))
                        uid == UInt32(0xffffffff) && continue
                        push!(local_idx[tid], global_start + UInt64(i - 1))
                        push!(local_uid[tid], UInt64(uid))
                    end
                else
                    @threads :static for i in 1:n
                        tid = threadid()
                        oi = Int(lp.offidx[i]) + 1
                        lid = off_lookup_id[oi]
                        lid == 0 && continue
                        cr = pack_confrot(lp.conf[i], lp.rot[i])
                        key = UInt64(cr) | off_suffix[oi]
                        uid = get(off_lookups[Int(lid)], key, UInt32(0xffffffff))
                        uid == UInt32(0xffffffff) && continue
                        push!(local_idx[tid], global_start + UInt64(i - 1))
                        push!(local_uid[tid], UInt64(uid))
                    end
                end
                @inbounds for tid in 1:nt
                    append!(out_idx, local_idx[tid])
                    append!(out_uid, local_uid[tid])
                end
            else
                if use_confrot_mask
                    @inbounds for i in 1:n
                        oi = Int(lp.offidx[i]) + 1
                        lid = off_lookup_id[oi]
                        lid == 0 && continue
                        cr = pack_confrot(lp.conf[i], lp.rot[i])
                        bitset_has(mask::Vector{UInt64}, cr) || continue
                        key = UInt64(cr) | off_suffix[oi]
                        uid = get(off_lookups[Int(lid)], key, UInt32(0xffffffff))
                        uid == UInt32(0xffffffff) && continue
                        push!(out_idx, global_start + UInt64(i - 1))
                        push!(out_uid, UInt64(uid))
                    end
                else
                    @inbounds for i in 1:n
                        oi = Int(lp.offidx[i]) + 1
                        lid = off_lookup_id[oi]
                        lid == 0 && continue
                        cr = pack_confrot(lp.conf[i], lp.rot[i])
                        key = UInt64(cr) | off_suffix[oi]
                        uid = get(off_lookups[Int(lid)], key, UInt32(0xffffffff))
                        uid == UInt32(0xffffffff) && continue
                        push!(out_idx, global_start + UInt64(i - 1))
                        push!(out_uid, UInt64(uid))
                    end
                end
            end
            end
        end

        if !isempty(out_idx)
            total_hits += UInt64(length(out_idx))
            if use_u32
                nout = length(out_idx)
                c1 = Vector{UInt32}(undef, nout)
                c2 = Vector{UInt32}(undef, nout)
                @inbounds for i in 1:nout
                    out_idx[i] <= UInt64(typemax(UInt32)) || error("Index exceeds UInt32 range")
                    out_uid[i] <= UInt64(typemax(UInt32)) || error("Unique index exceeds UInt32 range")
                    c1[i] = UInt32(out_idx[i])
                    c2[i] = UInt32(out_uid[i])
                end
                add_index2!(writer32::IndexFileWriter, c1, c2)
            else
                add_index64_2!(writer64::Index64FileWriter, out_idx, out_uid)
            end
        end
        global_start += UInt64(n)
    end
    if use_u32
        finish_index_writer!(writer32::IndexFileWriter)
    else
        finish_index64_writer!(writer64::Index64FileWriter)
    end
    return total_hits
end

function build_lookup_from_existing_unique_k!(
    unique_pairs::Vector{PairInfo},
    lookup_dir::String,
)::Tuple{UInt64, Dict{NTuple{3, Int16}, String}}
    mkpath(lookup_dir)
    raw_temp = use_raw_temp_files()
    tmp_paths = Dict{NTuple{3, Int16}, String}()
    uid_next = UInt64(0)
    for pair in unique_pairs
        lp = load_pair(pair)
        n = length(lp.conf)
        local_bufs = Dict{NTuple{3, Int16}, Vector{UInt8}}()
        @inbounds for i in 1:n
            oi = Int(lp.offidx[i]) + 1
            x = lp.abs_offsets[oi, 1]
            y = lp.abs_offsets[oi, 2]
            z = lp.abs_offsets[oi, 3]
            center = canonical_center(x, y, z)
            rx32 = Int32(x) - Int32(center[1])
            ry32 = Int32(y) - Int32(center[2])
            rz32 = Int32(z) - Int32(center[3])
            (rx32 >= -128 && rx32 <= 127 && ry32 >= -128 && ry32 <= 127 && rz32 >= -128 && rz32 <= 127) ||
                error("Canonical rebasing failed for $(pair.pose_path)")

            uid_next <= UInt64(typemax(UInt32)) || error("Unique-K index exceeds UInt32 range")
            uid = UInt32(uid_next)
            uid_next += UInt64(1)
            key = pack_rel_pose_key(lp.conf[i], lp.rot[i], Int8(rx32), Int8(ry32), Int8(rz32))
            buf = get!(local_bufs, center, UInt8[])
            p = length(buf) + 1
            resize!(buf, p + 11)
            write_keyuid_to_bytes!(buf, p, key, uid)
        end

        for (center, buf) in local_bufs
            path = get(tmp_paths, center, "")
            if isempty(path)
                path = joinpath(lookup_dir, center_key_filename(center))
                tmp_paths[center] = path
            end
            open(path, "a") do io
                write(io, buf)
            end
        end
    end

    lookup_paths = Dict{NTuple{3, Int16}, String}()
    if raw_temp
        for (center, path) in tmp_paths
            lookup_paths[center] = path
        end
    else
        for (center, path) in tmp_paths
            lookup_paths[center] = compress_file_zst_inplace(path)
        end
    end
    return uid_next, lookup_paths
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

function sum_pose_rows(pairs::Vector{PairInfo})::UInt64
    s = UInt64(0)
    for p in pairs
        s += UInt64(pose_row_count(p.pose_path))
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

function should_print_size_stats()::Bool
    raw = get(ENV, "CROCODILE_PRINT_SIZE_STATS", "0")
    return raw in ("1", "true", "TRUE", "yes", "YES")
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
    tmp_a = joinpath(tmp_root, "set-a-reorg")
    tmp_b = joinpath(tmp_root, "set-b-reorg")
    mkpath(tmp_a)
    mkpath(tmp_b)

    println("A (largest): $large_name")
    println("B (smallest): $small_name")
    println("threads: $(nthreads())")
    print_sizes = should_print_size_stats()

    lookup_dir = joinpath(tmp_root, "k-lookup")
    lookup_paths = Dict{NTuple{3, Int16}, String}()
    unique_k_count = UInt64(0)
    total_rows_large = UInt64(0)
    total_rows_small = UInt64(0)

    if cfg.reuse_unique_k_dir === nothing
        prefilter_shared_mask::Union{Nothing, Vector{UInt64}} = nothing
        if enable_global_confrot_prefilter()
            println("stage:start confrot-prefilter")
            t_stage = time()
            prefilter_shared_mask, uniq_confrot_a, shared_confrot = build_global_confrot_shared_mask_from_pairs(large_pairs, small_pairs)
            println("confrot prefilter (pre-canonical): unique-A=$uniq_confrot_a shared=$shared_confrot")
            println(@sprintf("stage:end confrot-prefilter %.3f", time() - t_stage))
        end

        println("stage:start canonicalization")
        t_stage = time()
        index_a = build_canonical_index!(large_pairs, tmp_a, prefilter_shared_mask)
        index_b = build_canonical_index!(small_pairs, tmp_b, prefilter_shared_mask)
        println(@sprintf("stage:end canonicalization %.3f", time() - t_stage))
        println("A poses indexed: $(index_a.total_poses), temp rewrite (raw est): $(fmt_bytes(index_a.temp_bytes))")
        println("B poses indexed: $(index_b.total_poses), temp rewrite (raw est): $(fmt_bytes(index_b.temp_bytes))")
        if print_sizes
            println("tmp size after canonicalization: $(fmt_bytes(dir_size_bytes(tmp_root)))")
        end

        println("stage:start unique-k")
        t_stage = time()
        unique_k_count, lookup_paths = build_unique_k_and_lookup!(
            index_a,
            index_b,
            cfg.output,
            lookup_dir,
            cfg,
            prefilter_shared_mask,
        )
        println(@sprintf("stage:end unique-k %.3f", time() - t_stage))
        total_rows_large = index_a.total_poses
        total_rows_small = index_b.total_poses
    else
        reuse_dir = cfg.reuse_unique_k_dir::String
        isdir(reuse_dir) || error("Unique-K directory not found: $reuse_dir")
        println("reuse unique-K from: $reuse_dir")
        reuse_pairs = discover_pose_pairs(reuse_dir)
        println("stage:start reuse-unique-k-lookup")
        t_stage = time()
        unique_k_count, lookup_paths = build_lookup_from_existing_unique_k!(reuse_pairs, lookup_dir)
        println(@sprintf("stage:end reuse-unique-k-lookup %.3f", time() - t_stage))
        total_rows_large = sum_pose_rows(large_pairs)
        total_rows_small = sum_pose_rows(small_pairs)
        println("A source rows: $total_rows_large")
        println("B source rows: $total_rows_small")
    end

    map_confrot_mask::Union{Nothing, Vector{UInt64}} = nothing
    if enable_map_confrot_prefilter()
        println("stage:start map-confrot-mask")
        t_stage = time()
        map_confrot_mask, map_confrot_uniq = build_map_confrot_mask_from_lookup_paths(lookup_paths)
        println("map confrot mask size: $map_confrot_uniq")
        println(@sprintf("stage:end map-confrot-mask %.3f", time() - t_stage))
    end

    println("stage:start map-a")
    t_stage = time()
    matched_a_rows = map_set_to_lookup!(
        large_pairs,
        lookup_paths,
        cfg.output,
        "ind-A",
        cfg;
        total_poses = total_rows_large,
        confrot_mask = map_confrot_mask,
    )
    println(@sprintf("stage:end map-a %.3f", time() - t_stage))

    println("stage:start map-b")
    t_stage = time()
    matched_b_rows = map_set_to_lookup!(
        small_pairs,
        lookup_paths,
        cfg.output,
        "ind-B",
        cfg;
        total_poses = total_rows_small,
        confrot_mask = map_confrot_mask,
    )
    println(@sprintf("stage:end map-b %.3f", time() - t_stage))
    println("matched A rows: $matched_a_rows")
    println("matched B rows: $matched_b_rows")
    println("unique-K rows: $unique_k_count")
    if print_sizes
        println("tmp size after matching: $(fmt_bytes(dir_size_bytes(tmp_root)))")
    end
    println("done")
    if print_sizes
        println("output size: $(fmt_bytes(dir_size_bytes(cfg.output)))")
    end

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
