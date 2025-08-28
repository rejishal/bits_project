from synthetic_data_generator import create_synthetic_banking_data

if __name__ == "__main__":
    df = create_synthetic_banking_data(n_samples=500000, random_state=42)
    df.to_csv("synthetic_banking_data.csv", index=False)
    print("Synthetic data saved to synthetic_banking_data.csv")