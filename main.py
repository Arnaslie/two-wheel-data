from two_wheel_data.data import load_bikes


def main():
    df = load_bikes()
    print(f"Shape: {df.shape}")
    print("First 5 records:")
    print(df.head())


if __name__ == "__main__":
    main()
