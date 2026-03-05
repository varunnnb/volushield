from src.encryption.pipeline import encrypt_volume
from src.metrics.performance import scalability_test

print("Running scalability test...")

results = scalability_test(
    encrypt_volume,
    sizes=[
        (64,64,64),
        (96,96,96),
        (128,128,64),
        (128,128,128)
    ]
)

print("\nScalability results:")
for r in results:
    print(r)