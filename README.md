# Machine Learning & Programming Practice

A collection of coursework and practice exercises organised by source and week. This repository is a folder of learning materials, not one packaged application.

## Collections

### Beeskilled

C programming assignments organised by week.

- `Beeskilled/week-1/Assignment/` — four small C exercises (`qn1.c`–`qn4.c`) and their checked-in Windows executables.
- `Beeskilled/week-2/Assignment/` — a C exercise (`qn1.c`).

Compile the source with a C compiler such as GCC rather than relying on the checked-in `.exe` files:

```bash
cd Beeskilled/week-1/Assignment
gcc qn1.c -o qn1
./qn1
```

On Windows, run the compiled program as `qn1.exe`.

### Nexis

Python practice and introductory data analysis.

- `Nexis/Week-1/music.py` — trains and evaluates a decision-tree classifier using `music.csv`.
- `Nexis/Week-1/vgsales.py` — practice analysis using `vgsales.csv`.
- `Nexis/Week-1/Assignment/titanic_cleaning.py` — fills missing values, encodes categories, plots age distribution and saves a cleaned Titanic dataset.

The scripts read their CSV files using paths relative to the script's working directory, so run each one from its containing folder.

Install the Python packages used by the current scripts:

```bash
python -m pip install pandas scikit-learn matplotlib seaborn
```

Run the music exercise:

```bash
cd Nexis/Week-1
python music.py
```

Run the Titanic cleaning exercise:

```bash
cd Nexis/Week-1/Assignment
python titanic_cleaning.py
```

## Repository Layout

```text
Machine-Learning/
├── Beeskilled/
│   ├── week-1/Assignment/
│   └── week-2/Assignment/
└── Nexis/
    └── Week-1/
        ├── Assignment/
        ├── music.csv
        ├── music.py
        ├── vgsales.csv
        └── vgsales.py
```

There is currently no shared Python environment file or project-wide test suite. Install dependencies for the exercises you intend to run.

## Author

**Yoson** · [GitHub](https://github.com/yo5on)
