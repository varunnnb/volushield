import oqs

message = b"VoluShield Test"

with oqs.Signature("Dilithium3") as signer:
    
    public_key = signer.generate_keypair()
    
    signature = signer.sign(message)

    print("Signature created")

    valid = signer.verify(message, signature, public_key)

    print("Signature valid:", valid)