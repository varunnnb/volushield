# src/crypto/dilithium.py

import oqs
import numpy as np
from typing import Tuple

class DilithiumCrypto:
    """
    Wrapper for CRYSTALS-Dilithium operations.
    Uses Dilithium3 (NIST security level 3, recommended for medical data).
    """
    
    ALGORITHM = "Dilithium3"
    
    def __init__(self):
        self.signer = oqs.Signature(self.ALGORITHM)
        self.public_key = None
        self.secret_key = None
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Dilithium3 keypair. Returns (public_key, secret_key)."""
        self.public_key = self.signer.generate_keypair()
        self.secret_key = self.signer.export_secret_key()
        return self.public_key, self.secret_key
    
    def sign_volume_hash(self, volume_hash: bytes) -> bytes:
        """Sign the SHA-512 hash of the medical volume."""
        signature = self.signer.sign(volume_hash)
        return signature
    
    def verify_signature(self, 
                         volume_hash: bytes, 
                         signature: bytes, 
                         public_key: bytes) -> bool:
        """Verify signature during decryption. Returns True if valid."""
        verifier = oqs.Signature(self.ALGORITHM)
        try:
            return verifier.verify(volume_hash, signature, public_key)
        except Exception:
            return False