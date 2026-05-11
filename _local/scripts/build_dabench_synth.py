"""Build a 15-question synthetic-data DABench-style benchmark.

Generates 3 random CSVs with non-standard schemas (zero memorization risk
for any LLM), then writes 15 questions with deterministic ground truth
computed via pandas. Output schema matches DABench so the existing
harness + grader work unchanged.
"""
import json, pathlib, hashlib
import numpy as np, pandas as pd

OUT_TABLES = pathlib.Path('bench/adaptive/dabench_synth_tables')
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_JSONL = pathlib.Path('bench/adaptive/dabench_synth_15.jsonl')
LAKE_TABLES = '/lakehouse/default/Files/fabric_rlm_dabench_synth_tables'

rng = np.random.default_rng(20260505)

# === Table 1: synthetic widget sales (n=2000) ===
n1 = 2000
regions = rng.choice(['Aurelia', 'Borvik', 'Cyndara', 'Drostal'], n1, p=[0.4, 0.3, 0.2, 0.1])
prices = rng.gamma(2.0, 25.0, n1).round(2)
units = rng.poisson(prices / 10).clip(0, 200)
discount = rng.beta(2, 5, n1).round(3)
revenue = (prices * units * (1 - discount)).round(2)
# inject ~5% nulls in discount
mask = rng.random(n1) < 0.05
discount_with_nulls = discount.astype(object)
discount_with_nulls[mask] = None
df1 = pd.DataFrame({
    'region': regions,
    'unit_price_kron': prices,
    'units_sold': units,
    'discount_frac': discount_with_nulls,
    'revenue_kron': revenue,
})
df1.to_csv(OUT_TABLES / 'widget_sales.csv', index=False)

# === Table 2: sensor telemetry (n=1500) ===
n2 = 1500
device = rng.choice(['DEV-A', 'DEV-B', 'DEV-C', 'DEV-D', 'DEV-E'], n2)
temp_c = rng.normal(42.0, 7.5, n2).round(2)
vibration = (rng.exponential(0.6, n2) + np.where(temp_c > 50, 0.8, 0)).round(3)
status = np.where(vibration > 1.5, 'fault', 'ok')
df2 = pd.DataFrame({
    'device_id': device,
    'temp_c': temp_c,
    'vibration_mm_s': vibration,
    'status': status,
})
df2.to_csv(OUT_TABLES / 'sensor_log.csv', index=False)

# === Table 3: fictional stock trades (n=3000) ===
n3 = 3000
ticker = rng.choice(['ZQRX', 'MTLP', 'KVDN', 'WBYZ', 'PLOM'], n3, p=[0.3, 0.25, 0.2, 0.15, 0.1])
qty = rng.integers(10, 1001, n3)
fill_price = (rng.normal(0, 1, n3) * 5 + 100).round(2)
side = rng.choice(['buy', 'sell'], n3, p=[0.55, 0.45])
df3 = pd.DataFrame({
    'ticker': ticker,
    'qty': qty,
    'fill_price': fill_price,
    'side': side,
})
df3.to_csv(OUT_TABLES / 'trades.csv', index=False)

# === Build 15 questions with computed ground truth ===
INSTR_TMPL = (
    "You are a data-analysis assistant. You have access to a Python interpreter "
    "with pandas/numpy/scipy installed. A CSV file is available on disk at:\n"
    "    {csv_path}\n"
    "Load the file with pandas (e.g. `pd.read_csv('{csv_path}')`) and answer the "
    "question below. Follow ALL constraints exactly.\n\n"
    "On the final line of your response, output your answer in EXACTLY this format:\n"
    "    {fmt}\n"
    "If the format lists multiple keys, output all of them on a single final line, "
    "comma-separated (e.g. `@key_a[v1], @key_b[v2]`). Use only ASCII brackets `[` and `]`. "
    "Do NOT add any text after the answer line.\n\n"
    "Question:\n{q}\n\nConstraints:\n{c}\n"
)

def fmt2(x):
    return f'{round(float(x), 2):.2f}'

questions = []

# Q1: count rows in widget_sales for region == 'Borvik'
v = int((df1['region'] == 'Borvik').sum())
questions.append(('widget_sales.csv',
    'How many rows have region equal to "Borvik" in the widget_sales table?',
    'No constraints beyond exact-match on the region string.',
    '@borvik_count[count_value]', [['borvik_count', str(v)]]))

# Q2: mean unit_price_kron for Aurelia, 2 decimals
v = df1.loc[df1['region']=='Aurelia','unit_price_kron'].mean()
questions.append(('widget_sales.csv',
    'What is the mean unit_price_kron for region "Aurelia"?',
    'Round to 2 decimal places.',
    '@aurelia_mean_price[value]', [['aurelia_mean_price', fmt2(v)]]))

# Q3: pearson correlation between unit_price_kron and revenue_kron, 2 decimals
v = df1[['unit_price_kron','revenue_kron']].corr(method='pearson').iloc[0,1]
questions.append(('widget_sales.csv',
    'Compute the Pearson correlation coefficient between unit_price_kron and revenue_kron.',
    'Use Pearson method. Round to 2 decimal places.',
    '@corr_price_revenue[value]', [['corr_price_revenue', fmt2(v)]]))

# Q4: median units_sold, integer
v = int(df1['units_sold'].median())
questions.append(('widget_sales.csv',
    'What is the median value of units_sold?',
    'Output as an integer.',
    '@median_units_sold[value]', [['median_units_sold', str(v)]]))

# Q5: number of non-null discount_frac
v = int(df1['discount_frac'].notna().sum())
questions.append(('widget_sales.csv',
    'How many rows have a non-null discount_frac value?',
    'Count non-null entries only.',
    '@nonnull_discount_count[value]', [['nonnull_discount_count', str(v)]]))

# Q6: total revenue per region, top region
totals = df1.groupby('region')['revenue_kron'].sum().sort_values(ascending=False)
top_region = totals.index[0]
top_value = totals.iloc[0]
questions.append(('widget_sales.csv',
    'Which region has the highest total revenue_kron, and what is that total?',
    'Round the total to 2 decimal places.',
    '@top_region[name], @top_revenue[value]',
    [['top_region', top_region], ['top_revenue', fmt2(top_value)]]))

# Q7: sensor_log fault rate
n_fault = int((df2['status']=='fault').sum())
rate = round(n_fault / len(df2), 3)
questions.append(('sensor_log.csv',
    'What fraction of rows in sensor_log have status equal to "fault"?',
    'Round to 3 decimal places.',
    '@fault_rate[value]', [['fault_rate', f'{rate:.3f}']]))

# Q8: mean temp_c by device, max
tmean = df2.groupby('device_id')['temp_c'].mean()
hot_device = tmean.idxmax()
hot_value = tmean.max()
questions.append(('sensor_log.csv',
    'Which device_id has the highest mean temp_c, and what is that mean?',
    'Round the mean to 2 decimal places.',
    '@hottest_device[name], @hottest_mean_temp[value]',
    [['hottest_device', hot_device], ['hottest_mean_temp', fmt2(hot_value)]]))

# Q9: pearson correlation between temp_c and vibration
v = df2[['temp_c','vibration_mm_s']].corr(method='pearson').iloc[0,1]
questions.append(('sensor_log.csv',
    'Compute the Pearson correlation between temp_c and vibration_mm_s.',
    'Use Pearson method. Round to 2 decimal places.',
    '@corr_temp_vib[value]', [['corr_temp_vib', fmt2(v)]]))

# Q10: 95th percentile of vibration_mm_s
v = float(np.percentile(df2['vibration_mm_s'], 95))
questions.append(('sensor_log.csv',
    'What is the 95th percentile of vibration_mm_s?',
    'Use linear interpolation. Round to 2 decimal places.',
    '@p95_vibration[value]', [['p95_vibration', fmt2(v)]]))

# Q11: trades total qty for ZQRX
v = int(df3.loc[df3['ticker']=='ZQRX','qty'].sum())
questions.append(('trades.csv',
    'What is the total qty traded for ticker "ZQRX"?',
    'Sum the qty column for rows where ticker == "ZQRX".',
    '@zqrx_total_qty[value]', [['zqrx_total_qty', str(v)]]))

# Q12: trades buy/sell ratio for MTLP, 2 decimals
m = df3[df3['ticker']=='MTLP']
b = (m['side']=='buy').sum()
s = (m['side']=='sell').sum()
ratio = round(b / s, 2) if s else 0
questions.append(('trades.csv',
    'For ticker "MTLP", compute the ratio of buy-side to sell-side trade counts.',
    'Output buy_count / sell_count rounded to 2 decimal places.',
    '@mtlp_buy_sell_ratio[value]', [['mtlp_buy_sell_ratio', fmt2(ratio)]]))

# Q13: mean fill_price overall, 2 decimals
v = df3['fill_price'].mean()
questions.append(('trades.csv',
    'What is the mean fill_price across all trades?',
    'Round to 2 decimal places.',
    '@mean_fill_price[value]', [['mean_fill_price', fmt2(v)]]))

# Q14: standard deviation of qty, 2 decimals (population stddev)
v = df3['qty'].std(ddof=0)
questions.append(('trades.csv',
    'Compute the population standard deviation (ddof=0) of qty across all trades.',
    'Use population standard deviation (divide by N, not N-1). Round to 2 decimal places.',
    '@qty_stddev[value]', [['qty_stddev', fmt2(v)]]))

# Q15: number of distinct tickers
v = int(df3['ticker'].nunique())
questions.append(('trades.csv',
    'How many distinct ticker values appear in the trades table?',
    'Count distinct ticker strings.',
    '@distinct_tickers[value]', [['distinct_tickers', str(v)]]))

# Write JSONL
out = []
for i, (csv, q, c, fmt, gold) in enumerate(questions, start=1):
    csv_path = f'{LAKE_TABLES}/{csv}'
    qid = f'DASYNTH_{i:03d}'
    prompt = INSTR_TMPL.format(csv_path=csv_path, fmt=fmt, q=q, c=c)
    out.append({
        'question_id': qid,
        'domain': 'data_analysis',
        'difficulty': 'easy',
        'template': 'DABENCH',
        'prompt': prompt,
        'answer': json.dumps(gold),
        'metadata': {'csv': csv, 'fmt': fmt},
    })

OUT_JSONL.write_text('\n'.join(json.dumps(r) for r in out) + '\n', encoding='utf-8')
print(f'wrote {OUT_JSONL} n={len(out)}')
print('tables:')
for p in sorted(OUT_TABLES.glob('*.csv')):
    print(f'  {p} {p.stat().st_size} bytes')
print()
print('--- gold table ---')
for r in out:
    print(f'  {r["question_id"]}: {r["answer"]}')
