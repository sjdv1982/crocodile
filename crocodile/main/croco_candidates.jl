module CrocoCandidates

#using Base.Threads
"""
Gives inconsistent results with threading: disable for now
"""

const VALUES_PER_CLUSTER = 16

@inline function last_word_mask(nrota::Int)
    rem = nrota % 64
    rem == 0 && return typemax(UInt64)
    return (UInt64(1) << rem) - UInt64(1)
end

function build_prefix_masks(membership::AbstractMatrix{UInt8})
    nclust, nrota = size(membership)
    nwords = (nrota + 63) >>> 6
    masks = zeros(UInt64, nwords, nclust, VALUES_PER_CLUSTER)

    @inbounds for c in 1:nclust
        row = @view membership[c, :]
        for j in 1:nrota
            val = row[j]
            word_idx = ((j - 1) >>> 6) + 1
            bit_shift = (j - 1) & 63
            masks[word_idx, c, Int(val) + 1] |= UInt64(1) << bit_shift
        end
    end

    @inbounds for value in 2:VALUES_PER_CLUSTER
        for c in 1:nclust
            prev = @view masks[:, c, value - 1]
            curr = @view masks[:, c, value]
            for word in 1:size(masks, 1)
                curr[word] |= prev[word]
            end
        end
    end

    return masks
end

function compute_candidates(
    membership::AbstractMatrix{UInt8},
    rmsd_upper::AbstractMatrix{UInt8},
    rmsd_lower::AbstractMatrix{UInt8};
)::Tuple{Vector{Int},Vector{Int}}
    nclust = size(membership, 1)
    nrota = size(membership, 2)
    size(rmsd_upper, 1) == nclust || throw(ArgumentError("upper matrix has incompatible size"))
    size(rmsd_lower, 1) == nclust || throw(ArgumentError("lower matrix has incompatible size"))
    size(rmsd_upper) == size(rmsd_lower) || throw(ArgumentError("upper and lower matrices must match"))
    nstruc = size(rmsd_upper, 2)

    prefix_masks = build_prefix_masks(membership)
    nwords = size(prefix_masks, 1)
    tail_mask = last_word_mask(nrota)
    nthreads = Threads.nthreads()
    nthreads = 1 ###

    candidate_buffers = [Vector{Int}(undef, nrota) for _ in 1:nthreads]
    word_buffers = [Vector{UInt64}(undef, nwords) for _ in 1:nthreads]
    candidate_counts = zeros(Int, nstruc)
    candidate_lists = Vector{Vector{Int}}(undef, nstruc)

    #Threads.@threads for n in 1:nstruc   ###
    #    tid = Threads.threadid()        ###
    for n in 1:nstruc
        tid = 1
        candidates = candidate_buffers[tid]
        candidate_words = word_buffers[tid]

        upper = @views rmsd_upper[:, n]
        lower = @views rmsd_lower[:, n]

        fill!(candidate_words, typemax(UInt64))
        candidate_words[end] = tail_mask

        @inbounds for c in 1:nclust
            low = Int(lower[c])
            high = Int(upper[c])
            high_slice = @view prefix_masks[:, c, high + 1]
            nonzero = false

            if low == 0
                for word in 1:nwords
                    filtered = candidate_words[word] & high_slice[word]
                    candidate_words[word] = filtered
                    nonzero = nonzero | (filtered != 0)
                end
            else
                low_slice = @view prefix_masks[:, c, low]
                for word in 1:nwords
                    filtered = candidate_words[word] & (high_slice[word] & ~low_slice[word])
                    candidate_words[word] = filtered
                    nonzero = nonzero | (filtered != 0)
                end
            end

            nonzero || break
        end

        candidate_count = 0
        @inbounds for word_idx in 1:nwords
            word = candidate_words[word_idx]
            offset = (word_idx - 1) << 6
            while word != 0
                tz = trailing_zeros(word)
                idx = offset + tz + 1
                if idx > nrota
                    break
                end
                candidate_count += 1
                candidates[candidate_count] = idx
                word &= word - UInt64(1)
            end
        end

        candidate_counts[n] = candidate_count
        if candidate_count == 0
            candidate_lists[n] = Int[]
        else
            stored = Vector{Int}(undef, candidate_count)
            copyto!(stored, 1, candidates, 1, candidate_count)
            candidate_lists[n] = sort(stored)
        end
    end

    total_candidates = sum(candidate_counts)
    all_candidates = Vector{Int}(undef, total_candidates)
    offset = 0
    @inbounds for n in 1:nstruc
        count = candidate_counts[n]
        if count > 0
            list = candidate_lists[n]
            copyto!(all_candidates, offset + 1, list, 1, count)
            offset += count
        end
    end

    return candidate_counts, all_candidates
end


function compute_candidates_DEFAULT(
    membership::AbstractMatrix{UInt8},
    rmsd_upper::AbstractMatrix{UInt8},
    rmsd_lower::AbstractMatrix{UInt8};
)::Tuple{Vector{Int},Vector{Int}}
    nclust = size(membership, 1)
    nrota = size(membership, 2)
    size(rmsd_upper, 1) == nclust || throw(ArgumentError("upper matrix has incompatible size"))
    size(rmsd_lower, 1) == nclust || throw(ArgumentError("lower matrix has incompatible size"))
    size(rmsd_upper) == size(rmsd_lower) || throw(ArgumentError("upper and lower matrices must match"))
    nstruc = size(rmsd_upper, 2)

    candidate_lists = Vector{Vector{Int}}(undef, nstruc)
    candidate_counts = zeros(Int, nstruc)

    candidate_mask = Vector{Bool}(undef, nrota)

    for n in 1:nstruc
        fill!(candidate_mask, true)
        upper = @views rmsd_upper[:, n]
        lower = @views rmsd_lower[:, n]

        @inbounds for c in 1:nclust
            mem = @views membership[c, :]
            up = upper[c]
            if up < 15
                @inbounds @simd for j in 1:nrota  #@turbo does not change
                    candidate_mask[j] &= (mem[j] <= up)
                end

            end

            low = lower[c]
            if low > 0
                @inbounds @simd for j in 1:nrota  #@turbo does not change
                    candidate_mask[j] &= (mem[j] >= low)
                end
            end
        end

        candidate_lists[n] = findall(candidate_mask)
        candidate_counts[n] = length(candidate_lists[n])
    end

    total_candidates = sum(candidate_counts)
    all_candidates = Vector{Int}(undef, total_candidates)
    offset = 0
    @inbounds for n in 1:nstruc
        count = candidate_counts[n]
        if count > 0
            list = candidate_lists[n]
            copyto!(all_candidates, offset + 1, list, 1, count)
            offset += count
        end
    end

    return candidate_counts, all_candidates

end

export compute_candidates, compute_candidates_DEFAULT

end
