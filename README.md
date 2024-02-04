## expenses-tracker
Tracks the expenses in a google sheet.

# Dependencies
The following python modules are required:

- gspread
- nuitka
- pandas
- scikit-learn
- scipy

# How to compile

    python -m nuitka --mingw64 --assume-yes-for-downloads --standalone --follow-imports --enable-plugin=tk-inter --include-data-dir=res=./res --remove-output --windows-icon-from-ico=res/wallet.ico --onefile-windows-splash-screen-image=res/wallet.ico --file-version=1.1 ExpensesTracker.py