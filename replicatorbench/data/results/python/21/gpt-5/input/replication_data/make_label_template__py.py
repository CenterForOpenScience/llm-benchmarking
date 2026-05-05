import os
import pandas as pd

DATA_DIR = "/app/data"
PART1_FILE = os.path.join(DATA_DIR, "Bischetti_Survey_Part1_deidentify.csv")
PART2_FILE = os.path.join(DATA_DIR, "Bischetti_Survey_Part2_deidentify.csv")
TEMPLATE_OUT = os.path.join(DATA_DIR, "pic_label_map_template.csv")


def main():
    if not os.path.exists(PART1_FILE) or not os.path.exists(PART2_FILE):
        raise FileNotFoundError(f"Place Bischetti_Survey_Part1_deidentify.csv and Bischetti_Survey_Part2_deidentify.csv in {DATA_DIR}")
    df1 = pd.read_csv(PART1_FILE, nrows=1)
    df2 = pd.read_csv(PART2_FILE, nrows=1)

    cols = list(df1.columns) + list(df2.columns)
    disturbing_cols = sorted({c for c in cols if 'disturbing' in c})

    names = [c.replace('disturbing', 'anote') for c in disturbing_cols]
    out = pd.DataFrame({'name': names, 'label': ''})

    out.to_csv(TEMPLATE_OUT, index=False)
    print(f"Wrote template with {len(out)} items to {TEMPLATE_OUT}. Fill 'label' with one of: covid-verbal, covid-meme, covid-strip, non-verbal.")


if __name__ == '__main__':
    main()
