from data_loader import create_dataloaders

print("Loading dataset...")

train_loader, val_loader, test_loader = create_dataloaders(
    root_dir="data/DUTS",
    img_size=128,
    batch_size=4
)

print("\n--- Loader Sizes ---")
print("Train batches:", len(train_loader))
print("Val batches:  ", len(val_loader))
print("Test batches: ", len(test_loader))

images, masks = next(iter(train_loader))

print("\n--- Batch Shapes ---")
print("Images shape:", images.shape)
print("Masks shape: ", masks.shape)
