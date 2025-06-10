"""
DGK Comparison Protocol Simulation

This module provides a non-encrypted simulation of the DGK secure comparison protocol [1],
demonstrating how two parties (A and B) can securely determine whether a secret integer m
held in additive shares is greater than a public integer x.

References:
  [1] Dai, Wei, Crystal Lin, and Dawn Song. “DGK: Efficient and Correct Comparison for
      Secure Protocols.” Crypto'10.
"""
import random
from typing import List, Tuple, Optional


def share_bits(m: int, bit_length: int, modulus: int) -> Tuple[List[int], List[int]]:
    """
    Secretly share the bit decomposition of m into two additive shares over Z_modulus.

    Args:
        m: The secret integer (0 <= m < 2**bit_length).
        bit_length: Number of bits to decompose.
        modulus: A prime-like modulus > bit_length + 2.

    Returns:
        A tuple of two lists (a_shares, b_shares), each of length bit_length,
        such that (a_shares[i] + b_shares[i]) mod modulus == i-th bit of m.
    """
    bits = [(m >> i) & 1 for i in range(bit_length)]
    a_shares, b_shares = [], []
    for bit in bits:
        a = random.randrange(modulus)
        b = (bit - a) % modulus
        a_shares.append(a)
        b_shares.append(b)
    return a_shares, b_shares


def bit_decompose(x: int, bit_length: int) -> List[int]:
    """Return the little-endian bit list of x up to bit_length bits."""
    return [(x >> i) & 1 for i in range(bit_length)]


def compute_w_shares(
    a_shares: List[int],
    b_shares: List[int],
    x_bits:   List[int],
    modulus:  int
) -> Tuple[List[int], List[int]]:
    """
    Compute shares of w_i = m_i XOR x_i for each bit position i.
    """
    alpha_w, beta_w = [], []
    for a, b, xi in zip(a_shares, b_shares, x_bits):
        # Using: m_i XOR x_i = m_i + x_i - 2*m_i*x_i over Z_modulus
        alpha_w.append((a + xi - 2*a*xi) % modulus)
        beta_w.append((b - 2*b*xi) % modulus)
    return alpha_w, beta_w


def compute_c_shares(
    a_shares: List[int],
    b_shares: List[int],
    x_bits:   List[int],
    alpha_w:  List[int],
    beta_w:   List[int],
    modulus:  int
) -> Tuple[List[int], List[int]]:
    """
    Compute shares of c_i = x_i - m_i + 1 + sum_{j=i+1}^{bit_length-1} w_j.
    """
    length = len(a_shares)
    alpha_c, beta_c = [], []
    # Precompute suffix sums to optimize to O(n)
    suffix_alpha = [0] * (length + 1)
    suffix_beta  = [0] * (length + 1)
    for i in range(length-1, -1, -1):
        suffix_alpha[i] = (alpha_w[i] + suffix_alpha[i+1]) % modulus
        suffix_beta[i]  = (beta_w[i]  + suffix_beta[i+1])  % modulus

    for i in range(length):
        # exclude w_i itself by using suffix at i+1
        sum_a = suffix_alpha[i+1]
        sum_b = suffix_beta[i+1]
        xi   = x_bits[i]
        alpha_c.append((xi - a_shares[i] + 1 + sum_a) % modulus)
        beta_c.append((    - b_shares[i] +   sum_b) % modulus)
    return alpha_c, beta_c


def reconstruct_and_test(
    alpha_c: List[int],
    beta_c:  List[int],
    modulus: int
) -> bool:
    """
    Reconstruct c_i values, apply random non-zero masks, permute,
    and test for existence of zero, which signals m > x.
    """
    combined = [(a + b) % modulus for a, b in zip(alpha_c, beta_c)]
    masked = []
    for val in combined:
        if val == 0:
            masked.append(0)
        else:
            r = random.randrange(1, modulus)
            masked.append((val * r) % modulus)
    random.shuffle(masked)
    return any(m == 0 for m in masked)


def compare(
    m: int,
    x: int,
    bit_length: Optional[int] = None,
    modulus:    Optional[int] = None
) -> bool:
    """
    High-level interface: Returns True iff secret m > public x under DGK simulation.

    Args:
        m: Secret integer.
        x: Public integer.
        bit_length: Optional bit-length; defaults to enough bits for max(m,x).
        modulus:    Optional modulus; defaults to bit_length*2 + 3.

    Returns:
        Boolean indicating if m > x.
    """
    # Determine defaults
    max_val = max(m, x)
    required_bits = max_val.bit_length()
    l = bit_length or required_bits
    u = modulus    or (2 * l + 3)

    a, b = share_bits(m, l, u)
    x_b = bit_decompose(x, l)
    aw, bw = compute_w_shares(a, b, x_b, u)
    ac, bc = compute_c_shares(a, b, x_b, aw, bw, u)
    return reconstruct_and_test(ac, bc, u)

# # Example invocation
# if __name__ == "__main__":
#     random.seed(42)
#     for m_val, x_val in [(5, 3), (3, 5), (7, 7), (11, 99), (100,  1000000), (1000000001, 9)]:
#         print(f"compare({m_val}, {x_val}) -> {compare(m_val, x_val)}")
