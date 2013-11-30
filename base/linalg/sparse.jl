## sparse matrix multiplication

function (*){TvA,TiA,TvB,TiB}(A::SparseMatrixCSC{TvA,TiA}, B::SparseMatrixCSC{TvB,TiB})
    Tv = promote_type(TvA, TvB)
    Ti = promote_type(TiA, TiB)
    A  = convert(SparseMatrixCSC{Tv,Ti}, A)
    B  = convert(SparseMatrixCSC{Tv,Ti}, B)
    A * B
end

(*){TvA,TiA}(A::SparseMatrixCSC{TvA,TiA}, X::BitArray{1}) = invoke(*, (SparseMatrixCSC, AbstractVector), A, X)
# In matrix-vector multiplication, the correct orientation of the vector is assumed.
function A_mul_B!(α::Number, A::SparseMatrixCSC, x::AbstractVector, β::Number, y::AbstractVector)
    A.n == length(x) || throw(DimensionMismatch(""))
    A.m == length(y) || throw(DimensionMismatch(""))
    for i = 1:A.m; y[i] *= β; end
    nzv = A.nzval
    rv = A.rowval
    for col = 1 : A.n
        αx = α*x[col]
        @inbounds for k = A.colptr[col] : (A.colptr[col+1]-1)
            y[rv[k]] += nzv[k]*αx
        end
    end
    y
end
*{TA,S,Tx}(A::SparseMatrixCSC{TA,S}, x::AbstractVector{Tx}) = A_mul_B!(1, A, x, 0, zeros(promote_type(TA,Tx), A.m))

*(X::BitArray{1}, A::SparseMatrixCSC) = invoke(*, (AbstractVector, SparseMatrixCSC), X, A)
# In vector-matrix multiplication, the correct orientation of the vector is assumed.
# XXX: this is wrong (i.e. not what Arrays would do)!!
function *{T1,T2}(X::AbstractVector{T1}, A::SparseMatrixCSC{T2})
    A.m==length(X) || throw(DimensionMismatch(""))
    Y = zeros(promote_type(T1,T2), A.n)
    nzv = A.nzval
    rv = A.rowval
    for col =1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        Y[col] += X[rv[k]] * nzv[k]
    end
    Y
end

*{TvA,TiA}(A::SparseMatrixCSC{TvA,TiA}, X::BitArray{2}) = invoke(*, (SparseMatrixCSC, AbstractMatrix), A, X)
function (*){TvA,TiA,TX}(A::SparseMatrixCSC{TvA,TiA}, X::AbstractMatrix{TX})
    mX, nX = size(X)
    A.n==mX || throw(DimensionMismatch(""))
    Y = zeros(promote_type(TvA,TX), A.m, nX)
    nzv = A.nzval
    rv = A.rowval
    colptr = A.colptr
    for multivec_col=1:nX, col=1:A.n
        Xc = X[col, multivec_col]
        @inbounds for k = colptr[col] : (colptr[col+1]-1)
           Y[rv[k], multivec_col] += nzv[k] * Xc
        end
    end
    Y
end

*{TvA,TiA}(X::BitArray{2}, A::SparseMatrixCSC{TvA,TiA}) = invoke(*, (AbstractMatrix, SparseMatrixCSC), X, A)
function *{TX,TvA,TiA}(X::AbstractMatrix{TX}, A::SparseMatrixCSC{TvA,TiA})
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch(""))
    Y = zeros(promote_type(TX,TvA), mX, A.n)
    for multivec_row=1:mX, col=1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        Y[multivec_row, col] += X[multivec_row, A.rowval[k]] * A.nzval[k]
    end
    Y
end

# Sparse matrix multiplication as described in [Gustavson, 1978]:
# http://www.cse.iitb.ac.in/graphics/~anand/website/include/papers/matrix/fast_matrix_mul.pdf
function *{Tv,Ti}(A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti})
    mA, nA = size(A)
    mB, nB = size(B)
    nA==mB || throw(DimensionMismatch(""))

    colptrA = A.colptr; rowvalA = A.rowval; nzvalA = A.nzval
    colptrB = B.colptr; rowvalB = B.rowval; nzvalB = B.nzval
    # TODO: Need better estimation of result space
    nnzC = min(mA*nB, length(nzvalA) + length(nzvalB))
    colptrC = Array(Ti, nB+1)
    rowvalC = Array(Ti, nnzC)
    nzvalC = Array(Tv, nnzC)

    @inbounds begin
        ip = 1
        xb = zeros(Ti, mA)
        x  = zeros(Tv, mA)
        for i in 1:nB
            if ip + mA - 1 > nnzC
                resize!(rowvalC, nnzC + max(nnzC,mA))
                resize!(nzvalC, nnzC + max(nnzC,mA))
                nnzC = length(nzvalC)
            end
            colptrC[i] = ip
            for jp in colptrB[i]:(colptrB[i+1] - 1)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in colptrA[j]:(colptrA[j+1] - 1)
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if xb[k] != i
                        rowvalC[ip] = k
                        ip += 1
                        xb[k] = i
                        x[k] = nzC
                    else
                        x[k] += nzC
                    end
                end
            end
            for vp in colptrC[i]:(ip - 1)
                nzvalC[vp] = x[rowvalC[vp]]
            end
        end
        colptrC[nB+1] = ip
    end

    splice!(rowvalC, colptrC[end]:length(rowvalC))
    splice!(nzvalC, colptrC[end]:length(nzvalC))

    # The Gustavson algorithm does not guarantee the product to have sorted row indices.
    Cunsorted = SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC)
    Ct = Cunsorted.'
    Ctt = Base.SparseMatrix.transpose!(Ct, SparseMatrixCSC(mA, nB, colptrC, rowvalC, nzvalC))
end

## solvers
import Base.LinAlg.BLAS: scal!

typealias AbstractVecOrMat{T} Union(AbstractVector{T}, AbstractMatrix{T})

function \{TA<:Number,TB<:Number}(A::SparseMatrixCSC{TA}, B::AbstractVecOrMat{TB})
    if istril(A) return cscsm!(false, 1, [], 1.0, [3,1,0,1,1], A.nzval, A.rowval, A.colptr, B, 0.0, zeros(eltype(B), size(B)), similar(B)) end
    if istriu(A) return cscsm!(false, 1, [], 1.0, [3,2,0,1,1], A.nzval, A.rowval, A.colptr, B, 0.0, zeros(eltype(B), size(B)), similar(B)) end
    return lufact(A)\B
end
function At_ldiv_B(A::SparseMatrixCSC, b::Vector)
    if istril(A) return cscsm!(true, 1, [], 1.0, [3,1,0,1,1], A.nzval, A.rowval, A.colptr, B, 0.0, zeros(eltype(B), size(B)), similar(B)) end
    if istriu(A) return cscsm!(true, 1, [], 1.0, [3,2,0,1,1], A.nzval, A.rowval, A.colptr, B, 0.0, zeros(eltype(B), size(B)), similar(B)) end
    return lufact(A).'\b
end

# Translation of cscsm.f from NIST sparse BLAS
function cscsm!(transa::Bool, unitd::Integer, dv::AbstractVector, α::Number, descra::Vector, val::AbstractVector, indx::AbstractVector, colptr::AbstractVector, B::AbstractVecOrMat, β::Number, C::AbstractVecOrMat, work::AbstractVector)
#--------------------------------------------------------------------
#         ------------ begin interface description ------------
#
#   Toolkit interface:
#   dcscsm -- compressed sparse column format triangular solve
#  
#   C <- alpha D inv(A) B + beta C    C <- alpha D inv(A') B + beta C
#   C <- alpha inv(A) D B + beta C    C <- alpha inv(A') D B + beta C
#   
#                                      ( ' indicates matrix transpose)
#  
#   Arguments:
#  
#   int transa  Indicates how to operate with the sparse matrix
#       0 : operate with matrix
#       1 : operate with transpose matrix
#  
#   int m   Number of rows in matrix A
#  
#   int n   Number of columns in matrix c
#  
#   int unitd   Type of scaling:
#                        1 : Identity matrix (argument dv[] is ignored)
#                        2 : Scale on left (row scaling)
#                        3 : Scale on right (column scaling)
#  
#   double alpha    Scalar parameter
#  
#   double beta     Scalar parameter
#  
#   int descra()    Descriptor argument.  Nine element integer array
#       descra(0) matrix structure
#           0 : general
#           1 : symmetric
#           2 : Hermitian
#           3 : Triangular
#           4 : Skew(Anti-Symmetric
#           5 : Diagonal
#       descra(1) upper/lower triangular indicator
#           1 : lower
#           2 : upper
#       descra(2) main diagonal type
#           0 : non-unit
#           1 : unit
#       descra(3) Array base 
#           0 : C/C++ compatible
#           1 : Fortran compatible
#       descra(4) repeated indices?
#           0 : unknown
#           1 : no repeated indices
#
#   double val()  scalar array of length nnz containing matrix entries.
#  
#   int indx()    integer array of length nnz containing row indices.
#
##   int pntrb()   integer array of length k such that pntrb(j)-pntrb(1)
##                 points to location in val of the first nonzero element 
##                 in column j.
##
##   int pntre()   integer array of length k such that pntre(j)-pntre(1)
##                 points to location in val of the last nonzero element 
##                 in column j.
### Modified to only one columns pointer vector colptr 
#
#   double b()    rectangular array with first dimension ldb.
#  
#   double c()    rectangular array with first dimension ldc.
#  
#   double work() scratch array of length lwork.  lwork should be at least
#                 max(m,n)
#  
#       ------------ end interface description --------------
#--------------------------------------------------------------------

#
#     Test input parameters:
#
    m = size(C, 1)
    n = size(C, 2)
    ldb = stride(B, 2)
    ldc = stride(C, 2)
    lwork = length(work)
    unitd >= 1 && unitd <= 3 || error("unitd must be either 1, 2 or 3")
    descra[1] == 3 || error("only triangular matrices supported")
    descra[2] >= 1 && descra[2] <= 2 || error("element two of descra must be 1 or 2")
    descra[3] >= 0 && descra[3] <= 1 || error("element three of descra must be 0 or 1")

    if α == 0.0
#       Quick return after scaling: 
        scal!(m*n, β, C, 1)
        return C
    end

    # transpose = true
        # if transa .eq. 0) transpose = 'N'

    scale = unitd == 1 ? 'N' : (unitd == 2 ? 'L' : 'R')
    diag = descra[3] == 0 ? 'N' : 'U'

#     Call kernel subroutine:

    if !transa
        uplo = descra[2] == 1 ? 'L' : 'U'
        cscsmk!(m, n, scale, dv, dv, α, uplo, diag, val, indx, colptr, B, ldb, β, C, ldc, work, lwork)
    else
        uplo = descra[2] == 2 ? uplo = 'L' : 'U'
        csrsmk!(m, n, scale, dv, dv, α, uplo, diag, val, indx, colptr, B, ldb, β, C, ldc, work, lwork)
    end
    return C
end

function cscsmk!(m::Integer, n::Integer, scale::Char, dvl::Vector, dvr::Vector, α::Number, uplo::Char, diag::Char, val::Vector, indx::Vector, colptr::Vector, B::AbstractVecOrMat, ldb::Integer, β::Number, C::AbstractVecOrMat, ldc::Integer, work::Vector, base::Integer)

    maxcache = 32000000
    cacheline = 4
#
#     Set some parameter flags
#
    if diag == 'U'
        unit = true
        nonunit = false
    else
        unit = false
        nonunit = true
    end
 
    left = false
    right = false
    if scale == 'L'
        left = true
    elseif scale == 'R'
        right = true
    elseif scale == 'B'
        left = true
        right = true
    end

    lower = false
    upper = false
    if uplo == 'L'
        lower = true
    else
        upper = true
    end

#
#     Calculate the number of columns for partitioning
#     b and c (rcol) by taking into acount the amount of 
#     cache required for each point column:
#     For rcol columns in block:
#         cachereq  = nnz + rcol*m*3 < maxcache
#          from val ---^      from c, b, and work
#
#     So,   rcol = (maxcache-nnz)/(m*3)
#
    rcol = min(div(maxcache - colptr[m+1], m*3), n)
    rhscols = div(n, rcol)
    if rhscols*rcol != n rhscols += 1 end

#     Now, loop through the rhscols block columns of c & b:
    @inbounds begin
    bl = 1
    for bl = 1:rhscols
        rhscolb = (bl - 1)*rcol + 1
        rhscole = rhscolb + rcol - 1
        if rhscole >= n rhscole = n end
        nb = rhscole - rhscolb + 1

#       Copy c into work

        copy!(work, 1, C, (bl-1)*ldc+1, ldc*rcol)

        if right

#         Assign dvr*b to c:

            for i = 1:m
                z = dvr[i]
                for j = rhscolb:rhscole
                   C[i,j] = z*B[i,j]
                end
            end
        else

#         Assign b to c:

            copy!(C, (rhscolb-1)*ldc + 1, B, ldb*(rhscolb-1)+1, m*nb)

        end

#       Loop through the rcol columns in this block:

        for l = rhscolb:rhscole

            if lower

#       Lower triangular:

                for j = 1:m
                    jb = colptr[j]
                    je = colptr[j+1]
                    z = C[j,l]
                    if nonunit
                        z /= val[jb]
                        C[j,l] = z
                    end
                    len = je - jb - 1
                    if unit && indx[jb] != j
                        len = je - jb
                        off = 0
                    else
                        off = 1
                    end
                    for i = 1:len
                        C[indx[i+jb+off-1],l] -= z*val[i+jb+off-1]
                    end
                end
        
            else

#       Upper triangular:

                for j = m:-1:1
                    jb = colptr[j]
                    je = colptr[j+1]
                    z = C[j,l]
                    if nonunit
                        z /= val[je-1]
                        C[j,l] = z
                    end
                    len = je - jb - 1
                    if unit && indx[je-1] != j len = je - jb end
                    for i = 1:len
                        C[indx[i+jb-1],l] -= z*val[i+jb-1]
                    end
                end
            end
        end

        if left
            for i = 1:m
                t = α*dvl[i]
                for j = rhscolb:rhscole
                    C[i,j] = t*C[i,j] + β*work[i,j]
                end
            end
        else
            for i = 1:m
                for j = rhscolb:rhscole
                    C[i,j] = α*C[i,j] + β*work[i,j]
                end
            end
        end

    end
    end
        
    return C
end

## triu, tril

function triu{Tv,Ti}(S::SparseMatrixCSC{Tv,Ti}, k::Integer)
    m,n = size(S)
    colptr = Array(Ti, n+1)
    nnz = 0
    for col = 1 : min(max(k+1,1), n+1)
        colptr[col] = 1
    end
    for col = max(k+1,1) : n
        for c1 = S.colptr[col] : S.colptr[col+1]-1
            S.rowval[c1] > col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    rowval = Array(Ti, nnz)
    nzval = Array(Tv, nnz)
    A = SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
    for col = max(k+1,1) : n
        c1 = S.colptr[col]
        for c2 = A.colptr[col] : A.colptr[col+1]-1
            A.rowval[c2] = S.rowval[c1]
            A.nzval[c2] = S.nzval[c1]
            c1 += 1
        end
    end
    A
end

function tril{Tv,Ti}(S::SparseMatrixCSC{Tv,Ti}, k::Integer)
    m,n = size(S)
    colptr = Array(Ti, n+1)
    nnz = 0
    colptr[1] = 1
    for col = 1 : min(n, m+k)
        l1 = S.colptr[col+1]-1
        for c1 = 0 : (l1 - S.colptr[col])
            S.rowval[l1 - c1] < col - k && break
            nnz += 1
        end
        colptr[col+1] = nnz+1
    end
    for col = max(min(n, m+k)+2,1) : n+1
        colptr[col] = nnz+1
    end
    rowval = Array(Ti, nnz)
    nzval = Array(Tv, nnz)
    A = SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
    for col = 1 : min(n, m+k)
        c1 = S.colptr[col+1]-1
        l2 = A.colptr[col+1]-1
        for c2 = 0 : l2 - A.colptr[col]
            A.rowval[l2 - c2] = S.rowval[c1]
            A.nzval[l2 - c2] = S.nzval[c1]
            c1 -= 1
        end
    end
    A
end

## diff

function sparse_diff1{Tv,Ti}(S::SparseMatrixCSC{Tv,Ti})
    m,n = size(S)
    m > 1 && return SparseMatrixCSC{Tv,Ti}(0, n, ones(n+1), Ti[], Tv[])
    colptr = Array(Ti, n+1)
    numnz = 2 * nnz(S) # upper bound; will shrink later
    rowval = Array(Ti, numnz)
    nzval = Array(Tv, numnz)
    numnz = 0
    colptr[1] = 1
    for col = 1 : n
        last_row = 0
        last_val = 0
        for k = S.colptr[col] : S.colptr[col+1]-1
            row = S.rowval[k]
            val = S.nzval[k]
            if row > 1
                if row == last_row + 1
                    nzval[numnz] += val
                    nzval[numnz]==zero(Tv) && (numnz -= 1)
                else
                    numnz += 1
                    rowval[numnz] = row - 1
                    nzval[numnz] = val
                end
            end
            if row < m
                numnz += 1
                rowval[numnz] = row
                nzval[numnz] = -val
            end
            last_row = row
            last_val = val
        end
        colptr[col+1] = numnz+1
    end
    splice!(rowval, numnz+1:length(rowval))
    splice!(nzval, numnz+1:length(nzval))
    return SparseMatrixCSC{Tv,Ti}(m-1, n, colptr, rowval, nzval)
end

function sparse_diff2{Tv,Ti}(a::SparseMatrixCSC{Tv,Ti})

    m,n = size(a)
    colptr = Array(Ti, max(n,1))
    numnz = 2 * nnz(a) # upper bound; will shrink later
    rowval = Array(Ti, numnz)
    nzval = Array(Tv, numnz)

    z = zero(Tv)

    colptr_a = a.colptr
    rowval_a = a.rowval
    nzval_a = a.nzval

    ptrS = 1
    colptr[1] = 1

    n == 0 || return SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)

    startA = colptr_a[1]
    stopA = colptr_a[2]

    rA = startA : stopA - 1
    rowvalA = rowval_a[rA]
    nzvalA = nzval_a[rA]
    lA = stopA - startA

    for col = 1:n-1
        startB, stopB = startA, stopA
        startA = colptr_a[col+1]
        stopA = colptr_a[col+2]

        rowvalB = rowvalA
        nzvalB = nzvalA
        lB = lA

        rA = startA : stopA - 1
        rowvalA = rowval_a[rA]
        nzvalA = nzval_a[rA]
        lA = stopA - startA

        ptrB = 1
        ptrA = 1

        while ptrA <= lA && ptrB <= lB
            rowA = rowvalA[ptrA]
            rowB = rowvalB[ptrB]
            if rowA < rowB
                rowval[ptrS] = rowA
                nzval[ptrS] = nzvalA[ptrA]
                ptrS += 1
                ptrA += 1
            elseif rowB < rowA
                rowval[ptrS] = rowB
                nzval[ptrS] = -nzvalB[ptrB]
                ptrS += 1
                ptrB += 1
            else
                res = nzvalA[ptrA] - nzvalB[ptrB]
                if res != z
                    rowval[ptrS] = rowA
                    nzval[ptrS] = res
                    ptrS += 1
                end
                ptrA += 1
                ptrB += 1
            end
        end

        while ptrA <= lA
            rowval[ptrS] = rowvalA[ptrA]
            nzval[ptrS] = nzvalA[ptrA]
            ptrS += 1
            ptrA += 1
        end

        while ptrB <= lB
            rowval[ptrS] = rowvalB[ptrB]
            nzval[ptrS] = -nzvalB[ptrB]
            ptrS += 1
            ptrB += 1
        end

        colptr[col+1] = ptrS
    end
    splice!(rowval, ptrS:length(rowval))
    splice!(nzval, ptrS:length(nzval))
    return SparseMatrixCSC{Tv,Ti}(m, n-1, colptr, rowval, nzval)
end

diff(a::SparseMatrixCSC, dim::Integer)= dim==1 ? sparse_diff1(a) : sparse_diff2(a)

## norm and rank

# TODO

# kron

function kron{Tv,Ti}(a::SparseMatrixCSC{Tv,Ti}, b::SparseMatrixCSC{Tv,Ti})
    numnzA = nnz(a)
    numnzB = nnz(b)

    numnz = numnzA * numnzB

    mA,nA = size(a)
    mB,nB = size(b)

    m,n = mA*mB, nA*nB

    colptr = Array(Ti, n+1)
    rowval = Array(Ti, numnz)
    nzval = Array(Tv, numnz)

    colptr[1] = 1

    colptrA = a.colptr
    colptrB = b.colptr
    rowvalA = a.rowval
    rowvalB = b.rowval
    nzvalA = a.nzval
    nzvalB = b.nzval

    col = 1

    @inbounds for j = 1:nA
        startA = colptrA[j]
        stopA = colptrA[j+1]-1
        lA = stopA - startA + 1

        for i = 1:nB
            startB = colptrB[i]
            stopB = colptrB[i+1]-1
            lB = stopB - startB + 1

            ptr_range = (1:lB) + (colptr[col]-1)

            colptr[col+1] = colptr[col] + lA * lB
            col += 1

            for ptrA = startA : stopA
                ptrB = startB
                for ptr = ptr_range
                    rowval[ptr] = (rowvalA[ptrA]-1)*mB + rowvalB[ptrB]
                    nzval[ptr] = nzvalA[ptrA] * nzvalB[ptrB]
                    ptrB += 1
                end
                ptr_range += lB
            end
        end
    end
    SparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end

## det, inv, cond

inv(A::SparseMatrixCSC) = throw(MemoryError("The inverse of a sparse matrix can often be dense and can cause the computer to run out of memory. If you are sure you have enough memory, please convert your matrix to a dense matrix."))

# TODO

## scale methods

# multiply by diagonal matrix as vector
function scale!{Tv,Ti}(C::SparseMatrixCSC{Tv,Ti}, A::SparseMatrixCSC, b::Vector)
    m, n = size(A)
    (n==length(b) && size(A)==size(C)) || throw(DimensionMismatch(""))
    numnz = nnz(A)
    C.colptr = convert(Array{Ti}, A.colptr)
    C.rowval = convert(Array{Ti}, A.rowval)
    C.nzval = Array(Tv, numnz)
    for col = 1:n, p = A.colptr[col]:(A.colptr[col+1]-1)
        C.nzval[p] = A.nzval[p] * b[col]
    end
    C
end

function scale!{Tv,Ti}(C::SparseMatrixCSC{Tv,Ti}, b::Vector, A::SparseMatrixCSC)
    m, n = size(A)
    (n==length(b) && size(A)==size(C)) || throw(DimensionMismatch(""))
    numnz = nnz(A)
    C.colptr = convert(Array{Ti}, A.colptr)
    C.rowval = convert(Array{Ti}, A.rowval)
    C.nzval = Array(Tv, numnz)
    for col = 1:n, p = A.colptr[col]:(A.colptr[col+1]-1)
        C.nzval[p] = A.nzval[p] * b[A.rowval[p]]
    end
    C
end

scale{Tv,Ti,T}(A::SparseMatrixCSC{Tv,Ti}, b::Vector{T}) =
    scale!(SparseMatrixCSC(size(A,1),size(A,2),Ti[],Ti[],promote_type(Tv,T)[]), A, b)

scale{T,Tv,Ti}(b::Vector{T}, A::SparseMatrixCSC{Tv,Ti}) =
    scale!(SparseMatrixCSC(size(A,1),size(A,2),Ti[],Ti[],promote_type(Tv,T)[]), b, A)
