#!/usr/bin/env python3
"""
Quantum Random Noise Generator
==============================

Forward-verifiable, one-way random generation.

Properties
----------
1. Randomness comes from a quantum RNG source when available.
2. Cryptographic expansion produces arbitrary amounts of output.
3. Context is cryptographically bound to the generated numbers.
4. A SHA-256 commitment provides a one-way proof.
5. Knowing the context + commitment does NOT reveal the numbers.
6. Knowing the numbers + context + nonce allows verification.

Protocol
--------
                    QUANTUM ENTROPY
                           |
                           v
                    +-------------+
                    | HKDF / CSPRNG|
                    +-------------+
                           |
                           v
                     RANDOM BYTES
                           |
                           +----------------+
                           |                |
                           v                v
                       NUMBERS           PROOF
                                           |
                              SHA256(context ||
                                     numbers ||
                                     nonce)

The proof commits to the result without revealing it.

Python standard library only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import struct
from dataclasses import dataclass
from typing import Any


# ============================================================
# Quantum entropy source
# ============================================================

class QuantumEntropySource:
    """
    Interface for a quantum RNG.

    A real hardware QRNG can be connected by replacing read()
    with the device's API.

    The default implementation uses the operating system's
    cryptographically secure entropy source.

    os.urandom() is NOT quantum RNG; it is a secure fallback.
    """

    def __init__(self, quantum_reader=None):
        self.quantum_reader = quantum_reader

    def read(self, n: int) -> bytes:
        if n <= 0:
            raise ValueError("n must be positive")

        if self.quantum_reader is not None:
            data = self.quantum_reader(n)

            if not isinstance(data, bytes):
                raise TypeError("quantum_reader must return bytes")

            if len(data) != n:
                raise ValueError(
                    "quantum_reader returned incorrect number of bytes"
                )

            return data

        # Secure fallback.
        return os.urandom(n)


# ============================================================
# HKDF
# ============================================================

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """
    HKDF-Extract using HMAC-SHA256.
    """
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    HKDF-Expand using HMAC-SHA256.
    """
    if length < 0:
        raise ValueError("length must be >= 0")

    hash_len = hashlib.sha256().digest_size

    if length > 255 * hash_len:
        raise ValueError("requested HKDF output is too large")

    output = bytearray()
    previous = b""

    for counter in range(1, (length + hash_len - 1) // hash_len + 1):
        previous = hmac.new(
            prk,
            previous + info + bytes([counter]),
            hashlib.sha256,
        ).digest()

        output.extend(previous)

    return bytes(output[:length])


# ============================================================
# Canonical context
# ============================================================

def canonical_context(context: Any) -> bytes:
    """
    Convert arbitrary JSON-compatible context into a canonical
    byte representation.

    This prevents differences in JSON formatting from changing
    the cryptographic proof.
    """
    return json.dumps(
        context,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# ============================================================
# Random-number generator
# ============================================================

@dataclass
class RandomProof:
    context: Any
    numbers: list[int]
    nonce: str
    commitment: str
    algorithm: str = "SHA256-HKDF"

    def export(self, include_numbers: bool = True) -> dict:
        result = {
            "algorithm": self.algorithm,
            "context": self.context,
            "nonce": self.nonce,
            "commitment": self.commitment,
        }

        if include_numbers:
            result["numbers"] = self.numbers

        return result


class QuantumOneWayRNG:
    """
    Quantum/secure entropy + cryptographic one-way proof.
    """

    def __init__(self, entropy_source: QuantumEntropySource | None = None):
        self.entropy = entropy_source or QuantumEntropySource()

    # --------------------------------------------------------
    # Internal seed construction
    # --------------------------------------------------------

    def _derive_stream(
        self,
        entropy: bytes,
        context_bytes: bytes,
        nonce: bytes,
        length: int,
    ) -> bytes:

        # Domain separation prevents accidental reuse.
        salt = hashlib.sha256(
            b"QUANTUM-RNG-SALT-v1" +
            nonce
        ).digest()

        ikm = (
            b"QUANTUM-RNG-IKM-v1" +
            entropy +
            context_bytes +
            nonce
        )

        prk = hkdf_extract(salt, ikm)

        info = (
            b"QUANTUM-RNG-OUTPUT-v1" +
            hashlib.sha256(context_bytes).digest()
        )

        return hkdf_expand(prk, info, length)

    # --------------------------------------------------------
    # Integer extraction
    # --------------------------------------------------------

    @staticmethod
    def _uniform_integer(
        stream: bytes,
        offset: int,
        minimum: int,
        maximum: int,
    ) -> tuple[int, int]:

        if minimum > maximum:
            raise ValueError("minimum must be <= maximum")

        range_size = maximum - minimum + 1

        if range_size <= 0:
            raise ValueError("invalid integer range")

        # Rejection sampling avoids modulo bias.
        bits = range_size.bit_length()
        bytes_needed = max(1, (bits + 7) // 8)

        limit = (1 << (bytes_needed * 8)) - (
            (1 << (bytes_needed * 8)) % range_size
        )

        while True:
            if offset + bytes_needed > len(stream):
                raise RuntimeError("entropy stream exhausted")

            chunk = stream[offset:offset + bytes_needed]
            offset += bytes_needed

            value = int.from_bytes(chunk, "big")

            if value < limit:
                return minimum + (value % range_size), offset

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    def generate(
        self,
        context: Any,
        count: int = 16,
        minimum: int = 0,
        maximum: int = 2**32 - 1,
    ) -> RandomProof:

        if count <= 0:
            raise ValueError("count must be positive")

        context_bytes = canonical_context(context)

        # Fresh entropy for every generation.
        entropy = self.entropy.read(64)

        # Public nonce prevents ambiguity between proof instances.
        nonce = secrets.token_bytes(32)

        # Generate substantially more bytes than normally needed.
        # Rejection sampling may consume additional bytes.
        stream = self._derive_stream(
            entropy=entropy,
            context_bytes=context_bytes,
            nonce=nonce,
            length=count * 16,
        )

        numbers = []
        offset = 0

        for _ in range(count):
            value, offset = self._uniform_integer(
                stream,
                offset,
                minimum,
                maximum,
            )

            numbers.append(value)

        # ----------------------------------------------------
        # One-way commitment
        # ----------------------------------------------------

        number_bytes = b"".join(
            struct.pack(">Q", n)
            for n in numbers
        )

        commitment_input = (
            b"QUANTUM-RNG-COMMITMENT-v1" +
            len(context_bytes).to_bytes(8, "big") +
            context_bytes +
            len(number_bytes).to_bytes(8, "big") +
            number_bytes +
            nonce
        )

        commitment = hashlib.sha256(
            commitment_input
        ).hexdigest()

        return RandomProof(
            context=context,
            numbers=numbers,
            nonce=nonce.hex(),
            commitment=commitment,
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    @staticmethod
    def verify(proof: RandomProof) -> bool:
        """
        Verify that the supplied numbers belong to the supplied
        context and commitment.
        """

        context_bytes = canonical_context(proof.context)

        nonce = bytes.fromhex(proof.nonce)

        number_bytes = b"".join(
            struct.pack(">Q", n)
            for n in proof.numbers
        )

        commitment_input = (
            b"QUANTUM-RNG-COMMITMENT-v1" +
            len(context_bytes).to_bytes(8, "big") +
            context_bytes +
            len(number_bytes).to_bytes(8, "big") +
            number_bytes +
            nonce
        )

        expected = hashlib.sha256(
            commitment_input
        ).hexdigest()

        return hmac.compare_digest(
            expected,
            proof.commitment,
        )


# ============================================================
# Demonstration
# ============================================================

def main():

    rng = QuantumOneWayRNG()

    context = {
        "experiment": "camera-noise-test",
        "session": 42,
        "operation": "generate",
        "timestamp": "2026-08-28T18:42:00+10:00",
    }

    proof = rng.generate(
        context=context,
        count=20,
        minimum=0,
        maximum=999999,
    )

    print()
    print("=" * 70)
    print("QUANTUM ONE-WAY RANDOM GENERATOR")
    print("=" * 70)

    print("\nContext:")
    print(json.dumps(proof.context, indent=2))

    print("\nGenerated numbers:")
    print(proof.numbers)

    print("\nNonce:")
    print(proof.nonce)

    print("\nOne-way commitment:")
    print(proof.commitment)

    print("\nVerification:")
    print(rng.verify(proof))

    # --------------------------------------------------------
    # Demonstrate tamper detection
    # --------------------------------------------------------

    tampered = RandomProof(
        context=proof.context,
        numbers=proof.numbers.copy(),
        nonce=proof.nonce,
        commitment=proof.commitment,
    )

    tampered.numbers[0] ^= 1

    print("\nAfter modifying one number:")
    print(rng.verify(tampered))

    # --------------------------------------------------------
    # Public proof
    # --------------------------------------------------------

    public_record = proof.export(include_numbers=False)

    print("\nPublic record:")
    print(json.dumps(public_record, indent=2))

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
