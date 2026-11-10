## expenses-tracker
Tracks the expenses in a Google sheet.

# Dependencies
The following python modules are required:

- gspread
- nuitka
- pandas
- scikit-learn
- scipy

# How to compile

    python -m nuitka --mingw64 --assume-yes-for-downloads --standalone --enable-plugin=tk-inter --enable-plugin=dill-compat --include-data-dir=res=./res --windows-console-mode=disable --remove-output --windows-icon-from-ico=res/wallet.ico --file-version=1.2.1 --product-version=1.2.1 --product-name="ExpensesTracker" --file-description="Expenses Management" --jobs=8 ExpensesTracker.py