"""Is education the way? — LUMS Science Symposium, Data Science category.

Analysis by ISL PA Team A of the 1994 US Census "Adult" dataset (48,842 rows).
The question: is education the strongest predictor of earning more than $50K,
does something else do more of the work, and for whom does education pay
more or less?

Usage:
    python lss_code.py                      # reads data/data.csv, saves figures to figures/
    python lss_code.py --data path/to.csv   # use a different CSV
    python lss_code.py --show               # also open each figure in a window
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

EDUCATION_ORDER = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
    'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors',
    'Masters', 'Prof-school', 'Doctorate',
]
EDUCATION_BANDS = {
    '< HS diploma': ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th'],
    'HS-grad': ['HS-grad'],
    'Some college / Assoc.': ['Some-college', 'Assoc-voc', 'Assoc-acdm'],
    'Bachelors': ['Bachelors'],
    'Postgraduate': ['Masters', 'Prof-school', 'Doctorate'],
}
EDUCATION_BAND_ORDER = list(EDUCATION_BANDS)
MARRIED_STATUSES = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
TOP_OCCUPATIONS = ['Exec-managerial', 'Prof-specialty']

# Variables ranked in Table 2: display name -> column
RANKED_VARIABLES = {
    'Marital status': 'marital-status',
    'Education': 'education',
    'Occupation': 'occupation',
    'Age band': 'age-band',
    'Hours band': 'hours-band',
    'Gender': 'gender',
    'Workclass': 'workclass',
    'Race': 'race',
    'Native country': 'native-country',
}


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def wilson_interval(successes, totals):
    """95% Wilson score interval, returned as percentages."""
    lower, upper = proportion_confint(successes, totals, alpha=0.05, method='wilson')
    return lower * 100, upper * 100


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------

def load_data(path):
    # skipinitialspace strips the leading spaces the raw census file has after each comma
    data = pd.read_csv(path, skipinitialspace=True)
    print(f"Loaded {len(data):,} rows x {data.shape[1]} columns from {path}")
    return data


def diagnose(data):
    section('Diagnostics: capital-gain leakage')
    # 99999 is the column's maximum; every row there is labelled >50K, so it is a
    # censoring code tied to the label rather than a real dollar amount.
    top_coded = data[data['capital-gain'] == 99999]
    print(f"Rows at capital-gain = 99999: {len(top_coded)}")
    print(f"Share of them earning >50K:   {(top_coded['income'] == '>50K').mean():.1%}")


def clean(data):
    data = data.replace('?', np.nan)

    # Built before anything is dropped; used for the descriptive marital comparison
    data['married'] = data['marital-status'].isin(MARRIED_STATUSES).astype(int)

    before = len(data)
    data = data.drop_duplicates()
    print(f"\nDropped {before - len(data)} duplicate rows; {len(data):,} remain")

    # fnlwgt is a survey sampling weight; capital-gain/loss leak the label
    data = data.drop(columns=['fnlwgt', 'capital-gain', 'capital-loss', 'relationship'])

    # Missing occupation is not random (see Table 1), so flag it and keep the rows
    data['occupation-status'] = np.where(data['occupation'].isna(), 'occupation missing', 'occupation present')
    data['occupation'] = data['occupation'].fillna('Unknown')

    data = data.dropna(subset=['educational-num'])

    band_of = {level: band for band, levels in EDUCATION_BANDS.items() for level in levels}
    data['education-band'] = data['education'].map(band_of)
    data['age-band'] = pd.cut(data['age'], bins=[16, 25, 35, 45, 55, 65, 90])
    data['hours-band'] = pd.qcut(data['hours-per-week'], q=5, duplicates='drop')

    data['gender'] = (data['gender'] == 'Male').astype(int)  # 1 = male, 0 = female
    data['income'] = (data['income'] == '>50K').astype(int)  # 1 = >50K, 0 = <=50K
    data['income-percent'] = data['income'] * 100
    return data


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figure_occupation_mix(data):
    """Figure 1: occupation mix within each education band."""
    counts = data['education-band'].value_counts()
    tick_labels = [f"{band}\nn={counts[band]:,}" for band in EDUCATION_BAND_ORDER]

    # Occupations stacked from lowest to highest >50K rate
    sorted_occupations = data.groupby('occupation')['income'].mean().sort_values().index
    shares = pd.crosstab(data['education-band'], data['occupation'], normalize='index') * 100
    shares = shares.reindex(index=EDUCATION_BAND_ORDER)[sorted_occupations]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.95, len(sorted_occupations)))
    shares.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7, edgecolor='white', linewidth=0.5)

    ax.set_title('Occupation mix by education band', fontsize=13, pad=15)
    ax.set_xlabel('Education band (n shown below each bar)', fontsize=11, labelpad=10)
    ax.set_ylabel('Share of people in each occupation (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(tick_labels)), tick_labels, rotation=0, fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Occupation (highest-paid on top)',
              bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    return fig


def figure_rate_by_education(data):
    """Figure 2: share earning >$50K at each education level."""
    rates = data.groupby('education')['income-percent'].mean().reindex(EDUCATION_ORDER)

    fig, ax = plt.subplots(figsize=(12, 6))
    rates.plot(kind='bar', ax=ax, color='#3B6FB6', width=0.8)
    ax.set_title('Share earning >$50K by education level', fontsize=13, pad=15)
    ax.set_ylabel('Share earning >$50K (%)')
    ax.set_xlabel('Education level')
    ax.set_xticks(range(len(EDUCATION_ORDER)), EDUCATION_ORDER, rotation=45, ha='right')
    fig.tight_layout()
    return fig


def figure_rate_by_education_and_gender(data):
    """Figure 3: the same rate split by gender."""
    plot_data = data.assign(Gender=data['gender'].map({0: 'Female', 1: 'Male'}))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_data, x='education', y='income-percent', hue='Gender',
                hue_order=['Female', 'Male'], palette={'Female': '#e377c2', 'Male': '#1f77b4'},
                order=EDUCATION_ORDER, errorbar=None, ax=ax)
    ax.set_title('Share earning >$50K by education level and gender', fontsize=13, pad=15)
    ax.set_ylabel('Share earning >$50K (%)')
    ax.set_xlabel('Education level')
    ax.set_xticks(range(len(EDUCATION_ORDER)), EDUCATION_ORDER, rotation=45, ha='right')
    ax.legend(title='Gender', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    return fig


def figure_education_gradient(data, prof_vs_doc_p):
    """Figure 4: the education gradient with 95% Wilson intervals."""
    stats = (data.groupby(['educational-num', 'education'])['income']
             .agg(n='count', n_50K='sum').reset_index())
    stats['rate_pct'] = stats['n_50K'] / stats['n'] * 100
    lower, upper = wilson_interval(stats['n_50K'], stats['n'])
    base_rate = data['income'].mean() * 100

    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.errorbar(stats['education'], stats['rate_pct'],
                    yerr=[stats['rate_pct'] - lower, upper - stats['rate_pct']],
                    fmt='-o', color='#1f497d', ecolor='#8c96a0', capsize=4,
                    elinewidth=1.5, markeredgewidth=1.5, linewidth=2)
        ax.axhline(base_rate, color='#c0504d', linestyle='--', linewidth=1.2)
        ax.text(0.5, base_rate + 1.5, f"base rate {base_rate:.1f}%", color='#c0504d', fontsize=11)

        ax.set_title('The education gradient', fontsize=13, pad=15)
        ax.set_ylabel('% earning >$50K', fontsize=12)
        ax.set_ylim(-2, 85)
        ax.tick_params(axis='x', labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha='right', fontsize=11)
        sns.despine(ax=ax)

        fig.text(0.5, 0.01,
                 f"Bars are 95% Wilson intervals. Prof-school vs Doctorate: p = {prof_vs_doc_p:.2f}, "
                 "not distinguishable.", ha='center', fontsize=11, color='#555555')
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# --------------------------------------------------------------------------
# Tables and tests
# --------------------------------------------------------------------------

def table_missing_occupation(data):
    """Table 1: are people with no recorded occupation like everyone else?"""
    section('Table 1: missing occupation is not random')
    table = data.groupby('occupation-status').agg(
        n=('income', 'size'),
        rate_50K=('income', lambda x: x.mean() * 100),
        mean_age=('age', 'mean'),
        mean_education=('educational-num', 'mean'),
        mean_hours=('hours-per-week', 'mean'),
    ).round({'rate_50K': 2, 'mean_age': 1, 'mean_education': 2, 'mean_hours': 1})
    table.columns = ['n', 'rate >$50K (%)', 'mean age', 'mean education-num', 'mean hours']
    print(table.to_string())

    groups = data.groupby('occupation-status')['income'].agg(['sum', 'count'])
    z, p = proportions_ztest(groups['sum'], groups['count'])
    print(f"\nTwo-proportion z-test (missing vs present): z = {z:.2f}, p = {p:.1e}")


def table_variable_ranking(data):
    """Table 2: rank every variable by how much of the >50K variation it explains."""
    section('Table 2: which variables explain the most? (eta-squared)')
    overall_rate = data['income'].mean()
    total_squared_deviation = ((data['income'] - overall_rate) ** 2).sum()

    rows = []
    for name, col in RANKED_VARIABLES.items():
        stats = data.groupby(data[col].dropna(), observed=False)['income'].agg(n='count', rate='mean')
        # eta^2 = sum over categories of n * (category rate - overall rate)^2, over total squared deviation
        explained = (stats['n'] * (stats['rate'] - overall_rate) ** 2).sum()
        rows.append({
            'variable': name,
            'categories': len(stats),
            'range_pp': (stats['rate'].max() - stats['rate'].min()) * 100,
            'eta2_pct': explained / total_squared_deviation * 100,
        })

    table = pd.DataFrame(rows)
    table['rank by range'] = table['range_pp'].rank(ascending=False).astype(int)
    table['rank by η²'] = table['eta2_pct'].rank(ascending=False).astype(int)
    table = table.sort_values('rank by η²')
    table['range'] = table['range_pp'].map('{:.1f}pp'.format)
    table['η²'] = table['eta2_pct'].map('{:.2f}%'.format)
    print(table[['variable', 'categories', 'range', 'rank by range', 'η²', 'rank by η²']].to_string(index=False))


def table_rate_by_education(data):
    """Table 3: rate at each education level with 95% Wilson intervals."""
    section('Table 3: >$50K rate by education level')
    stats = (data.groupby(['educational-num', 'education'])['income']
             .agg(n='count', n_50K='sum').reset_index())
    lower, upper = wilson_interval(stats['n_50K'], stats['n'])
    stats['rate'] = (stats['n_50K'] / stats['n'] * 100).map('{:.1f}%'.format)
    stats['95% CI (Wilson)'] = [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(lower, upper)]
    stats = stats.rename(columns={'education': 'education level', 'n_50K': 'n >50K'})
    print(stats[['education level', 'n', 'n >50K', 'rate', '95% CI (Wilson)']].to_string(index=False))

    pair = stats.set_index('education level').loc[['Prof-school', 'Doctorate']]
    _, p = proportions_ztest(pair['n >50K'], pair['n'])
    print(f"\nProf-school vs Doctorate, two-proportion z-test: p = {p:.2f}")
    return p


def table_occupation_by_band(data):
    """Table 4: share in the two best-paying occupations vs income rate, per education band."""
    section('Table 4: education band vs top occupations (occupation recorded only)')
    # This table is about occupation, so rows with no recorded occupation are left out here
    data = data[data['occupation-status'] == 'occupation present']
    bands = pd.Categorical(data['education-band'], categories=EDUCATION_BAND_ORDER, ordered=True)
    table = data.assign(top_occ=data['occupation'].isin(TOP_OCCUPATIONS).astype(int)).groupby(
        bands, observed=False).agg(n=('income', 'count'), occ_rate=('top_occ', 'mean'), income_rate=('income', 'mean'))
    table.index.name = 'education band'
    table['% in Exec-managerial or Prof-specialty'] = (table['occ_rate'] * 100).map('{:.1f}%'.format)
    table['>$50K rate'] = (table['income_rate'] * 100).map('{:.1f}%'.format)
    print(table[['n', '% in Exec-managerial or Prof-specialty', '>$50K rate']].to_string())


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data/data.csv', help='path to the Adult census CSV')
    parser.add_argument('--out', default='figures', help='folder to save figures in')
    parser.add_argument('--show', action='store_true', help='display figures as well as saving them')
    args = parser.parse_args()

    data = load_data(args.data)
    diagnose(data)
    data = clean(data)

    table_missing_occupation(data)
    table_variable_ranking(data)

    # Grouped only after Table 2, which ranks native country at full detail
    data['native-country'] = np.where(data['native-country'].isna(), 'Unknown',
                                      np.where(data['native-country'] == 'United-States', 'US', 'Non-US'))

    prof_vs_doc_p = table_rate_by_education(data)
    table_occupation_by_band(data)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        '1_occupation_mix_by_education.png': figure_occupation_mix(data),
        '2_rate_by_education.png': figure_rate_by_education(data),
        '3_rate_by_education_and_gender.png': figure_rate_by_education_and_gender(data),
        '4_education_gradient.png': figure_education_gradient(data, prof_vs_doc_p),
    }
    for filename, fig in figures.items():
        fig.savefig(out / filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved {len(figures)} figures to {out}/")

    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
