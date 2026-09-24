device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([1, 2, 3]).to(device)
print(x)