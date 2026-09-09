"""Check candidate questions for degeneracy before they enter the bank.

A question only discriminates if its reference value and its hazard value
differ by more than the grader tolerance. q_service_avg_talk taught this the
hard way: its reference (14.9936) and hazard (14.994) were within tolerance,
so it scored 3/3 in both arms and measured nothing.

Offline: pandas only, no model calls, no library query compiler.
"""

from __future__ import annotations

import pathlib

import pandas as pd

DATA = pathlib.Path("stage3_data")
TOLERANCE = 0.01  # relative


def frames():
    names = [
        "olist_orders",
        "olist_order_items",
        "olist_order_payments",
        "bakehouse_sales_transactions",
        "callcenter_call_log",
    ]
    return {n: pd.read_parquet(DATA / f"{n}.parquet") for n in names}


def candidates(f):
    orders = f["olist_orders"]
    items = f["olist_order_items"]
    pay = f["olist_order_payments"]
    bake = f["bakehouse_sales_transactions"]
    calls = f["callcenter_call_log"]

    out = {}

    # ---- ecommerce -------------------------------------------------
    out["q_ecom_avg_payment"] = (
        pay.payment_value.sum() / pay.order_id.nunique(),
        pay.payment_value.mean(),
    )
    out["q_ecom_order_count"] = (items.order_id.nunique(), len(items))
    out["q_ecom_freight_share"] = (
        items.freight_value.sum() / (items.price.sum() + items.freight_value.sum()) * 100,
        (items.freight_value / (items.price + items.freight_value)).mean() * 100,
    )
    per_order_rows = pay.groupby("order_id").size()
    out["q_ecom_multi_installment_share"] = (
        (per_order_rows > 1).sum() / pay.order_id.nunique() * 100,
        pay.order_id.isin(per_order_rows[per_order_rows > 1].index).sum() / len(pay) * 100,
    )
    delivered = set(orders.loc[orders.order_status == "delivered", "order_id"])
    di = items[items.order_id.isin(delivered)]
    out["q_ecom_delivered_avg_items"] = (
        len(di) / di.order_id.nunique(),
        len(di) / len(delivered),
    )

    # ---- food retail -----------------------------------------------
    rev_prod = bake.groupby("product").totalPrice.sum().sort_values(ascending=False)
    out["q_retail_top_product"] = (rev_prod.iloc[0], rev_prod.iloc[1])
    out["q_retail_avg_per_customer"] = (
        bake.totalPrice.sum() / bake.customerID.nunique(),
        bake.totalPrice.mean(),
    )
    tx_per_cust = bake.groupby("customerID").size()
    out["q_retail_repeat_customer_share"] = (
        (tx_per_cust > 1).sum() / bake.customerID.nunique() * 100,
        bake.customerID.isin(tx_per_cust[tx_per_cust > 1].index).sum() / len(bake) * 100,
    )
    rev_fr = bake.groupby("franchiseID").totalPrice.sum().sort_values(ascending=False)
    out["q_retail_top_franchise"] = (rev_fr.iloc[0], rev_fr.iloc[1])
    qty_prod = bake.groupby("product").quantity.sum().sort_values(ascending=False)
    out["q_retail_top_product_qty"] = (qty_prod.iloc[0], qty_prod.iloc[1])

    # ---- service ops -----------------------------------------------
    out["q_service_avg_talk"] = (
        calls.Talk_Time.mean(),
        calls.groupby("Agent_ID").Talk_Time.mean().mean(),
    )
    out["q_service_abandoned_rate"] = (
        calls.Is_Abandoned.sum() / len(calls) * 100,
        calls.Is_Abandoned.sum() / (len(calls) - calls.Is_Abandoned.sum()) * 100,
    )
    out["q_service_avg_handle_time"] = (
        (calls.Talk_Time + calls.After_Call_Work_Time).mean(),
        calls.Talk_Time.mean(),
    )
    scored = calls[calls.Quality_Scored == 1]
    out["q_service_satisfied_rate_quality"] = (
        scored.Customer_Satisfied.sum() / len(scored) * 100,
        calls.Customer_Satisfied.sum() / len(calls) * 100,
    )
    ct = calls.Call_Type.value_counts()
    out["q_service_top_type_share"] = (
        ct.iloc[0] / len(calls) * 100,
        ct.iloc[1] / len(calls) * 100,
    )
    return out


def main():
    rows = candidates(frames())
    print(f"{'question':36} {'reference':>14} {'hazard':>14} {'rel_diff':>9}  verdict")
    bad = []
    for qid, (ref, haz) in rows.items():
        ref, haz = float(ref), float(haz)
        rel = abs(ref - haz) / abs(ref) if ref else float("inf")
        ok = rel > TOLERANCE * 2
        if not ok:
            bad.append(qid)
        print(
            f"{qid:36} {ref:>14.4f} {haz:>14.4f} {rel:>8.2%}  "
            f"{'OK' if ok else 'DEGENERATE'}"
        )
    print()
    if bad:
        print("EXCLUDE (hazard indistinguishable from reference):")
        for qid in bad:
            print("  -", qid)
    else:
        print("all candidates discriminate")


if __name__ == "__main__":
    main()
