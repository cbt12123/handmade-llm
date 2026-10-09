from knowledge import build_index

if __name__ == '__main__':
    data = build_index()
    print(f"Indexed {len(data['sources'])} files and {len(data['chunks'])} chunks.")
