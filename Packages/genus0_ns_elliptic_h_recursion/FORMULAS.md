# Formula and implementation contract

The full derivation and TeX-ready equations are in docs/machine_note.tex.
All labels in this document are one-based; Python tuple indices are
zero-based. There is a single normalization of H throughout.

## Sphere/pillow relation

Place d_1,...,d_n at (0,z,t_1,...,t_(n-4),1,infinity), and let m=n-3.
Use c=3/2+3Q^2, Q=b+1/b and q=exp(i*pi*tau)=prod_i p_i. The human-note
normalized elliptic integral is K(z)=2F1(1/2,1/2;1;z), so
tau=i*K(1-z)/K(z). The torus coordinate w has periods 2*pi and 2*pi*tau.

    F_(n,epsilon) = Lambda_n^(c)
                    * prod_i varrho_i^(h_i-c/24)
                    * C_NS(q) * H_(n,epsilon)

    Lambda_n^(c) = z^(c/24-d_1-d_2)
                  * (1-z)^(c/24-d_2-d_(n-1))
                  * theta_3(q)^[c/2-4(d_1+d_2+d_(n-1)+d_n)
                                -2 sum_(j=1)^(m-1) d_(j+2)]
                  * prod_(j=1)^(m-1) [t_j(1-t_j)(t_j-z)]^(-d_(j+2)/2)

    C_NS(q) = theta_3(q^2) / prod_(k>=1)(1-q^(2k))^(3/4)
            = 1 + (11/4)q^2 + (93/32)q^4 + ...

The powers above use the ordered real sheet. varrho=(16q) if m=1,
and (4p_1,p_2,...,p_(m-1),4p_m) if m>1. The interior p_i are not
multiplied by four. The factor is the actual-c geometric prefactor,
not the bosonic effective-charge prefactor.

The regular product is a cap matrix-element factor. It is not the
ordinary torus descendant character prod(1+q^(2k-1))/(1-q^(2k)).

## Exact coordinate products

Define U_j=prod_(i<=j)p_i, V_j=prod_(i>j)p_i, U_j*V_j=q. Then

    z = 16q * prod_(k>=1) [(1+q^(2k))/(1+q^(2k-1))]^8

    t_j = 4V_j (1+U_j)^2
          * prod_(k>=1) [(1+q^(2k))/(1+q^(2k-1))]^4
          * prod_(k>=1) [ (1+U_j*q^(2k))(1+V_j*q^(2k-1))
                         /((1+U_j*q^(2k-1))(1+V_j*q^(2k-2))) ]^2

The p_i are positive on the real OPE sheet; this fixes the collision
phases inherited from w. The implementation numerically inverts these
exact products, not a low-order expansion of the map.

## Null data and parity

    h_(r,s) = [Q^2-(r*b+s/b)^2]/8
    ell_(r,s) = r*s/2,  pi_(r,s) = r*s modulo 2
    r,s >= 1, r+s even

The fixed-c h-recursion includes (r,s)=(1,1). The r>=2 restriction of
moving-c recursion does not apply here.

    A_(r,s)^c = (1/2) prod [ (u*b+v/b)/sqrt(2) ]^(-1)

Here u=1-r,...,r and v=1-s,...,s, with u+v even; omit (0,0),(r,s).

For lambda_x^2=Q^2-8x, the ordered human-note fusion polynomial is

    P_(r,s)^alpha(x,y;c)
       = prod [(lambda_x-lambda_y+u*b+v/b)/(2*sqrt(2))]
              [(lambda_x+lambda_y+u*b+v/b)/(2*sqrt(2))].

The product runs over u=1-r,3-r,...,r-1 and v=1-s,3-s,...,s-1 with
u+v-r-s congruent to 2(1-alpha) modulo 4. The runtime pairs opposite
lattice sites, eliminating momentum square roots from this polynomial.
For (1,1), A=1/2, P^0(x,y)=y-x, P^1=1.

For edge parity epsilon_i, vertex labels are

    alpha = (epsilon_1, epsilon_1 XOR epsilon_2, ...,
             epsilon_(m-1) XOR epsilon_m, epsilon_m).

The three-form conventions are rho_0(primary,primary,primary)=1 and
rho_1(primary,G_(-1/2)primary,primary)=1. Odd signs must not be removed.

## Recursion

At a pole on edge k define, for every j,

    h_j^* = h_j-h_k+h_(r,s)
    h'_j = h_j^* + ell_(r,s)*delta_(j,k)
    epsilon'_j = epsilon_j XOR [pi_(r,s)*delta_(j,k)].

Adjacent ordered weight pairs are

    left  = (d_1,d_2)              if k=1
            (h_(k-1)^*,d_(k+1))   otherwise
    right = (d_n,d_(n-1))          if k=m
            (h_(k+1)^*,d_(k+2))   otherwise.

    R_(r,s)^(k,epsilon) = (-1)^(r*s) A_(r,s)^c
                         P_(r,s)^alpha_k(left;c)
                         P_(r,s)^alpha_(k+1)(right;c)

    H_(n,epsilon)(h)
       = delta_(epsilon,0)
         + sum_k sum_(r,s)
           varrho_k^(r*s/2) R_(r,s)^(k,epsilon)(h^*)
           /(h_k-h_(r,s)) * H_(n,epsilon')(h').

At the next recursive call, differences are recomputed from h', rather
than frozen to the original values. c, external weights and coordinates
remain fixed. C_NS cancels from the residue kernel because it is
independent of the internal weights.

The assumed regular part before dividing by C_NS is

    lim_(h_1 -> infinity, h_i-h_1 fixed) G_(n,epsilon)
       = delta_(epsilon,0) C_NS(q).

This general-n large-weight statement is a proposal supported by the
recorded finite-order checks, not an independent cap Ward-identity proof.

## Coefficient implementation

Use twice-level integers N_i=2*ell_i. Include sum_i N_i <= 2*order.
A pole contributes only when r*s<=N_k, and its child key is N-r*s*e_k.
The numerical coefficient multiplier from varrho_k is 4^(r*s) for
four points, 2^(r*s) at a higher-point cap, and 1 at an interior edge.

The runtime stores recursive weight differences as exact integer
half-level shifts. This avoids identifying distinct recursion states
through rounded floating-point weight keys. Every evaluated coefficient
belongs to the unique sector epsilon_i=N_i modulo 2.

## Independent c-recursion check

The Belavin--Geiko comparison is done in plane ratios
x=(z/t_1,t_1/t_2,...,t_(m-1)), or x=z at four points. Their published
q_i are these ratios in reversed ordering, not the elliptic nomes.
Their charge is c_BG=(2/3)c: both moving pole and Jacobian are converted.

Its seed is global osp(1|2) sewing; its child changes c to c_(r,s)(h_k)
and shifts only h_k by r*s/2. This is independent of the common-weight
elliptic construction. The complete coordinate and prefactor conversion
is applied before comparing H coefficients. See machine-note sections
6.5--6.6 for the equation-(3.18) indexing caveat and numerical results.
