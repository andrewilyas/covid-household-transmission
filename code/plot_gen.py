from urllib.request import urlopen
import pandas as pd
import yss_analyze
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import root
from scipy.integrate import quad
from scipy.stats import distributions
from scipy.special import loggamma, gamma
from argparse import ArgumentParser
import torch as ch
import re

from pathlib import Path
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

parser = ArgumentParser()
parser.add_argument('--out-dir', required=True)
parser.add_argument('--data-dir', required=True)
args = parser.parse_args() 

def get_weibull_params(mu, sigma):
    """
    Exploiting the identity:
      1/2 log(gamma(1+2/k)) - log(gamma(1+1/k)) = 1/2 log(mu^2 + sig^2) - log(mu)
    """
    f_targ = 0.5 * np.log(sigma**2 + mu**2) - np.log(mu)
    f = lambda x: 0.5 * loggamma(1+2/x) - loggamma(1+1/x) - f_targ
    # Find zero of f(), starting guess for k = 1e-4
    k = root(f, 1e-4).x[0]
    lam = mu / gamma(1 + 1/k)
    return (k, lam)

def calc_R0(k, lam, r, T):
    integrand = lambda t: distributions.weibull_min.pdf(t, c=k, scale=lam) * np.exp(-r * t)
    integral, err = quad(integrand, 0, T) 
    return 1 / integral

k, lam = get_weibull_params(5.0, 1.9)
calc_R0_prefilled = lambda x: calc_R0(k, lam, x, 1000)

lockdowns = pd.concat([pd.read_csv(str(Path(args.data_dir) / 'state_lockdown_dates.csv')), 
    pd.read_csv(str(Path(args.data_dir) / 'country_lockdown_dates.csv'))]).set_index('Place')
del lockdowns['Unnamed: 0']

def process_df(df, key):
    df = df.groupby(key).agg('sum')
    [df.__delitem__(c) for c in df.columns if not re.match('\d+/\d+/20', c)]
    df = df.reset_index()
    df = df.melt(id_vars=[key], var_name="date", 
        value_name="death").rename(columns={ key: 'region' })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

DATA_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/"
states_df = pd.read_csv(urlopen(f"{DATA_BASE}/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"))
countries_df = pd.read_csv(urlopen(f"{DATA_BASE}/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"))

states_df = process_df(states_df, 'Province_State')
countries_df = process_df(countries_df, 'Country/Region')
countries_df = countries_df[countries_df['region'] != 'Georgia']
deaths_df = pd.concat([states_df, countries_df]).set_index('region')

STATES = set(states_df['region'])
COUNTRIES = set(countries_df['region'])

df = pd.merge(lockdowns.loc[STATES | COUNTRIES], deaths_df.loc[STATES | COUNTRIES], 
                                             left_index=True, right_index=True)
for col in ['Date enacted', 'Date lifted', 'date']:
    df[col] = pd.to_datetime(df[col])

def calculate_growth(series, thresh=20, start=5):
    series = np.array(series)
    days = np.arange(0, series.shape[0])
    if ((series < thresh).all()):
        return (None, None, None)
    diff = np.diff(np.insert(series, 0, 0))
    keep = (series >= start) & (diff > 0)
    series = series[keep]
    days = days[keep]

    if (days.shape[0] < 3):
        return (None, None, None)
    model = yss_analyze.ExponentialGrowthRateEstimator(family='NegativeBinomial', alpha=0.10)
    try:
        model.fit(day=days, cases=series)
    except:
        return (None, None, None)
    print('True: %s' % np.diff(series))
    print('Pred: %s' % np.round(model.fitted_glm.mu, decimals=1))
    print(model.summary())

    est = model.growth_rate()
    low, high = model.growth_rate_confint()
    return tuple(map(calc_R0_prefilled, (low, est, high)))

def calculate_growth_df(_df):
    l_df = _df.sort_values('date').reset_index().groupby('index').agg(list)
    growth_rates = l_df['death'].apply(calculate_growth)
    _df['R0'] = growth_rates.apply(lambda x: x[1])
    _df['R0_low'] = growth_rates.apply(lambda x: x[0])
    _df['R0_high'] = growth_rates.apply(lambda x: x[2])
    return _df.reset_index().rename(columns={
        'index': 'region'
    }).groupby('region')[['R0','R0_low','R0_high']].mean().dropna()


pre_lockdown = df[df['date'] < df['Date enacted'] + timedelta(days=7)]
post_lockdown = df[(df['date'] > df['Date enacted'] + timedelta(days=14))
                    & (df['date'] < df['Date lifted'] + timedelta(days=7))]

pre_lockdown = calculate_growth_df(pre_lockdown)
post_lockdown = calculate_growth_df(post_lockdown)

agg_df = pd.merge(pre_lockdown, post_lockdown, left_index=True, right_index=True)

matplotlib.style.use("ggplot")
rc('font', **{'family': 'Computer Modern', 
                'serif': ['Computer Modern'], 
                'size': 144})
rc('text', usetex=True)

for i, places in enumerate([STATES, COUNTRIES]):
    r_data = agg_df.loc[places].dropna()
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.bar(np.arange(len(r_data))-0.2, r_data['R0_x'], width=0.4, 
        color=sns.color_palette("tab10")[0], label="Pre-lockdown",
        yerr=[r_data['R0_x'] - r_data['R0_low_x'], r_data['R0_high_x'] - r_data['R0_x']])
    ax.bar(np.arange(len(r_data))+0.2, r_data['R0_y'], width=0.4, 
        color=sns.color_palette("tab10")[1], label="Post-lockdown",
        yerr=[r_data['R0_y'] - r_data['R0_low_y'], r_data['R0_high_y'] - r_data['R0_y']])

    ax.set_xticks(range(len(r_data)))
    ax.set_xticklabels(r_data.index)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(Path(args.out_dir) / f'r_bars_{i}.pdf'), bbox_inches='tight')

mob_df = pd.read_csv(urlopen('https://www.dropbox.com/s/y6oncsdrgphlzo6/google.csv?raw=1'))
mob_df = mob_df[(mob_df['Sector'] == 'Residential') & mob_df['County'].isna()]
country_mobs = mob_df[mob_df['State'].isna()].set_index('Country')[['Date', 'Percent Change']]
state_mobs = mob_df[(mob_df['Country'] == 'United States')].set_index('State')[['Date', 'Percent Change']]
mobs = pd.concat([country_mobs, state_mobs])
mobs = pd.merge(mobs, lockdowns, left_index=True, right_index=True)
for col in ['Date enacted', 'Date lifted', 'Date']:
    mobs[col] = pd.to_datetime(mobs[col])

mobs = mobs[(mobs['Date'] >= mobs['Date enacted']) & (mobs['Date'] < mobs['Date enacted'] + timedelta(days=7))]
avg_residential_mobility = mobs.reset_index().groupby('index').agg({'Percent Change': 'mean'})

rc('font', **{'family': 'Computer Modern', 
                'serif': ['Computer Modern'], 
                'size': 12})
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
Rh_mult = (1 + avg_residential_mobility['Percent Change'] / 100) * 0.3
Rh_mult = Rh_mult.loc[agg_df.index]
ax.scatter(agg_df['R0_y'] - Rh_mult, Rh_mult, 
    color=sns.color_palette("tab10", 2)[1], label='Post-lockdown')
ax.scatter(agg_df['R0_x'] - 0.3, np.ones_like(Rh_mult) * 0.3,
    color=sns.color_palette("tab10", 2)[0], label='Pre-lockdown')
x = np.arange(0, 5.0, 0.01)
y = np.arange(0.2, 0.45, 0.01)
X, Y = np.meshgrid(x, y)

Z = X + Y
cs = ax.contour(X, Y, Z, colors='gray', alpha=0.5, 
    levels=[0.8, 2.4, 4.0])
plt.clabel(cs, inline=1, fontsize=10, manual=[(0.5, 0.25), (2.0, 0.25), (3.75, 0.25)])
ax.set(
    xlim=(0, None),
    ylabel='Within-household reproduction ($R_h$)',
    xlabel='Community reproduction ($R_c$)'
)
ax.legend()
fig.savefig(str(Path(args.out_dir) / 'rc_vs_rh.pdf'), bbox_inches='tight')

Rh_share_pre = Rh_mult / agg_df['R0_x']
Rh_share_post = Rh_mult / agg_df['R0_y']
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.hist(Rh_share_pre, bins=np.linspace(0,1,25), alpha=0.5, 
    color=sns.color_palette("tab10", 2)[0], label='Pre-lockdown')
ax.hist(Rh_share_post, bins=np.linspace(0,1,25), alpha=0.5, 
    color=sns.color_palette("tab10", 2)[1], label='Post-lockdown')
ax.legend()
ax.set(xlabel='$R_h / R$',
       ylabel='Number of states')
fig.savefig(str(Path(args.out_dir) / 'rh_share.pdf'), bbox_inches='tight')