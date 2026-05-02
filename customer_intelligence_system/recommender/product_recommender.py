"""
Product Recommender Module
Generates personalized product recommendations using Market Basket Analysis
(Association Rules / Apriori) combined with RFM customer segments.

How it works:
1. Builds a customer-product matrix from raw transactions
2. Mines frequent itemsets using the Apriori algorithm
3. Generates association rules (if customer bought X → recommend Y)
4. Filters recommendations per customer based on their RFM segment
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import RECOMMENDER_CONFIG, HEADER_LINE, SUBHEADER_LINE


class ProductRecommender:
    """
    Market Basket Analysis engine for personalized product recommendations.

    Uses the Apriori algorithm to discover association rules from transaction
    data, then filters and ranks recommendations per customer based on their
    RFM segment strategy.

    Attributes:
        df_transactions (pd.DataFrame): Cleaned transaction-level data
        customer_segments (pd.DataFrame): RFM segment data per customer
        rules (pd.DataFrame): Mined association rules
        recommendations (dict): Final recommendations per CustomerID
    """

    def __init__(self):
        """Initialize recommender with empty state."""
        self.df_transactions = None
        self.customer_segments = None
        self.product_lookup = {}       # StockCode → Description mapping
        self.basket_matrix = None      # Binary customer-product matrix
        self.frequent_itemsets = None
        self.rules = None
        self.recommendations = {}

    # =========================================================================
    # STEP 1 — LOAD DATA
    # =========================================================================

    def load_transactions(self, df_clean: pd.DataFrame):
        """
        Load the cleaned transaction DataFrame produced by CustomerAnalytics.

        Args:
            df_clean: Cleaned DataFrame with columns:
                      CustomerID, InvoiceNo, StockCode, Description, Quantity,
                      UnitPrice, InvoiceDate, TotalPrice

        Returns:
            self (for method chaining)
        """
        print(HEADER_LINE)
        print("PRODUCT RECOMMENDER — LOADING TRANSACTIONS")
        print(HEADER_LINE)

        self.df_transactions = df_clean.copy()

        # Build product lookup: StockCode → Description
        if 'Description' in self.df_transactions.columns:
            self.product_lookup = (
                self.df_transactions
                .dropna(subset=['Description'])
                .drop_duplicates('StockCode')
                .set_index('StockCode')['Description']
                .to_dict()
            )

        print(f"\n✓ Transactions loaded")
        print(f"  Rows        : {len(self.df_transactions):,}")
        print(f"  Customers   : {self.df_transactions['CustomerID'].nunique():,}")
        print(f"  Products    : {self.df_transactions['StockCode'].nunique():,}")
        print(f"  Invoices    : {self.df_transactions['InvoiceNo'].nunique():,}")

        return self

    def load_customer_segments(self, customer_results: pd.DataFrame):
        """
        Load the enriched customer table (output of intelligence_pipeline).

        Args:
            customer_results: DataFrame with at least CustomerID and Segment columns.

        Returns:
            self
        """
        self.customer_segments = customer_results[
            ['CustomerID', 'Segment', 'Customer_Type', 'Priority_Flag']
        ].copy()
        print(f"\n✓ Customer segments loaded ({len(self.customer_segments):,} customers)")
        return self

    # =========================================================================
    # STEP 2 — BUILD BASKET MATRIX
    # =========================================================================

    def build_basket_matrix(self):
        """
        Build a binary invoice-product matrix needed for Apriori.

        Each row = one invoice.
        Each column = one StockCode.
        Value = 1 if that product appeared in that invoice, else 0.

        Returns:
            self
        """
        print(f"\n{HEADER_LINE}")
        print("BUILDING BASKET MATRIX")
        print(HEADER_LINE)

        min_support_count = RECOMMENDER_CONFIG['min_support_count']

        # Keep only products that appear in at least min_support_count invoices
        product_counts = (
            self.df_transactions
            .groupby('StockCode')['InvoiceNo']
            .nunique()
        )
        popular_products = product_counts[
            product_counts >= min_support_count
        ].index

        df_filtered = self.df_transactions[
            self.df_transactions['StockCode'].isin(popular_products)
        ]

        print(f"\n  Products after min-support filter : {len(popular_products):,} "
              f"(removed {self.df_transactions['StockCode'].nunique() - len(popular_products):,} rare products)")

        # Build binary invoice × product matrix
        self.basket_matrix = (
            df_filtered
            .groupby(['InvoiceNo', 'StockCode'])['Quantity']
            .sum()
            .unstack(fill_value=0)
            .clip(upper=1)          # binarize: any quantity → 1
        )

        print(f"  Basket matrix shape               : {self.basket_matrix.shape}")
        print(f"  Density                           : "
              f"{(self.basket_matrix.values.sum() / self.basket_matrix.size) * 100:.2f}%")

        return self

    # =========================================================================
    # STEP 3 — MINE FREQUENT ITEMSETS (APRIORI)
    # =========================================================================

    def _apriori(self, min_support: float):
        """
        Pure-Python Apriori implementation (no mlxtend dependency).

        Mines frequent 1-itemsets and 2-itemsets only (pairs).
        Pairs are sufficient for generating A → B style rules.

        Args:
            min_support: Minimum support threshold (0–1).

        Returns:
            frequent_pairs (dict): {frozenset({A, B}): support_value}
        """
        n_transactions = len(self.basket_matrix)
        products = list(self.basket_matrix.columns)
        matrix = self.basket_matrix.values  # numpy for speed

        product_idx = {p: i for i, p in enumerate(products)}

        # --- Frequent 1-itemsets ---
        item_counts = matrix.sum(axis=0)  # count per product
        freq_1 = {
            frozenset([products[i]]): item_counts[i] / n_transactions
            for i in range(len(products))
            if item_counts[i] / n_transactions >= min_support
        }

        print(f"\n  Frequent 1-itemsets : {len(freq_1):,}")

        # --- Frequent 2-itemsets (pairs) ---
        freq_items = [list(fs)[0] for fs in freq_1]
        freq_2 = {}

        for a, b in combinations(freq_items, 2):
            ia, ib = product_idx[a], product_idx[b]
            co_occur = np.logical_and(matrix[:, ia], matrix[:, ib]).sum()
            support = co_occur / n_transactions
            if support >= min_support:
                freq_2[frozenset([a, b])] = support

        print(f"  Frequent 2-itemsets : {len(freq_2):,}")

        return freq_1, freq_2

    def mine_association_rules(self):
        """
        Run Apriori and derive association rules with confidence & lift.

        Rule format: antecedent → consequent
        For pairs {A, B}:
            Rule 1: A → B   (confidence = support(A,B) / support(A))
            Rule 2: B → A   (confidence = support(A,B) / support(B))

        Returns:
            self
        """
        print(f"\n{HEADER_LINE}")
        print("MINING ASSOCIATION RULES")
        print(HEADER_LINE)

        min_support    = RECOMMENDER_CONFIG['min_support']
        min_confidence = RECOMMENDER_CONFIG['min_confidence']
        min_lift       = RECOMMENDER_CONFIG['min_lift']

        print(f"\n  Min support    : {min_support}")
        print(f"  Min confidence : {min_confidence}")
        print(f"  Min lift       : {min_lift}")

        freq_1, freq_2 = self._apriori(min_support)

        if not freq_2:
            print("\n⚠️  No frequent pairs found. Consider lowering min_support in config.")
            self.rules = pd.DataFrame()
            return self

        # Generate rules from pairs
        rows = []
        for pair, pair_support in freq_2.items():
            items = list(pair)
            a, b = items[0], items[1]
            sup_a = freq_1.get(frozenset([a]), 0)
            sup_b = freq_1.get(frozenset([b]), 0)

            if sup_a > 0:
                conf_ab = pair_support / sup_a
                lift_ab = conf_ab / sup_b if sup_b > 0 else 0
                if conf_ab >= min_confidence and lift_ab >= min_lift:
                    rows.append({
                        'antecedent': a,
                        'consequent': b,
                        'support': round(pair_support, 4),
                        'confidence': round(conf_ab, 4),
                        'lift': round(lift_ab, 4)
                    })

            if sup_b > 0:
                conf_ba = pair_support / sup_b
                lift_ba = conf_ba / sup_a if sup_a > 0 else 0
                if conf_ba >= min_confidence and lift_ba >= min_lift:
                    rows.append({
                        'antecedent': b,
                        'consequent': a,
                        'support': round(pair_support, 4),
                        'confidence': round(conf_ba, 4),
                        'lift': round(lift_ba, 4)
                    })

        self.rules = (
            pd.DataFrame(rows)
            .sort_values('lift', ascending=False)
            .reset_index(drop=True)
        )

        print(f"\n✓ Association rules mined")
        print(f"  Total rules : {len(self.rules):,}")

        if not self.rules.empty:
            print(f"\n  Top 5 rules by lift:")
            print(f"  {'Antecedent':<15} → {'Consequent':<15}  "
                  f"{'Support':>8}  {'Confidence':>10}  {'Lift':>6}")
            print(f"  {'-'*65}")
            for _, row in self.rules.head(5).iterrows():
                ant_name = self.product_lookup.get(row['antecedent'], row['antecedent'])[:14]
                con_name = self.product_lookup.get(row['consequent'], row['consequent'])[:14]
                print(f"  {ant_name:<15} → {con_name:<15}  "
                      f"{row['support']:>8.4f}  {row['confidence']:>10.4f}  {row['lift']:>6.2f}")

        return self

    # =========================================================================
    # STEP 4 — GENERATE RECOMMENDATIONS PER CUSTOMER
    # =========================================================================

    def _get_segment_strategy(self, segment: str) -> dict:
        """
        Return recommendation strategy parameters for a given RFM segment.

        Champions and Loyal customers get premium/complementary product recs.
        At-Risk and Hibernating customers get high-confidence "safe" recs
        based on their own purchase history to reactivate familiarity.

        Args:
            segment: RFM segment string (e.g. 'Champions', 'At_Risk')

        Returns:
            dict with keys: max_recs, strategy_label
        """
        strategies = RECOMMENDER_CONFIG['segment_strategies']
        return strategies.get(segment, strategies['default'])

    def generate_recommendations(self):
        """
        Generate top-N product recommendations for every customer.

        Algorithm per customer:
        1. Get the set of products this customer has already purchased.
        2. Look up all rules where antecedent ∈ customer's purchases.
        3. Filter out products the customer already owns.
        4. Rank remaining candidates by lift × confidence.
        5. Keep top-N based on their segment strategy.

        Returns:
            self
        """
        print(f"\n{HEADER_LINE}")
        print("GENERATING CUSTOMER RECOMMENDATIONS")
        print(HEADER_LINE)

        if self.rules is None or self.rules.empty:
            print("⚠️  No rules available — skipping recommendation generation.")
            return self

        # Build customer → set of purchased StockCodes
        customer_products = (
            self.df_transactions
            .groupby('CustomerID')['StockCode']
            .apply(set)
            .to_dict()
        )

        # Build antecedent → list of (consequent, lift, confidence) lookup
        rule_lookup = defaultdict(list)
        for _, row in self.rules.iterrows():
            rule_lookup[row['antecedent']].append({
                'product': row['consequent'],
                'lift': row['lift'],
                'confidence': row['confidence'],
                'score': row['lift'] * row['confidence']
            })

        no_history = 0
        no_rules   = 0
        success    = 0

        all_customers = (
            self.customer_segments['CustomerID'].tolist()
            if self.customer_segments is not None
            else list(customer_products.keys())
        )

        for customer_id in all_customers:
            purchased = customer_products.get(customer_id, set())

            if not purchased:
                no_history += 1
                self.recommendations[customer_id] = []
                continue

            # Collect candidate recommendations
            candidates = {}
            for product in purchased:
                for rec in rule_lookup.get(product, []):
                    p = rec['product']
                    if p not in purchased:  # don't recommend already-bought items
                        if p not in candidates or rec['score'] > candidates[p]['score']:
                            candidates[p] = rec

            if not candidates:
                no_rules += 1
                self.recommendations[customer_id] = []
                continue

            # Get segment-specific max_recs
            if self.customer_segments is not None:
                seg_row = self.customer_segments[
                    self.customer_segments['CustomerID'] == customer_id
                ]
                segment = seg_row['Segment'].values[0] if len(seg_row) else 'default'
            else:
                segment = 'default'

            strategy = self._get_segment_strategy(segment)
            max_recs = strategy['max_recs']

            # Rank and keep top-N
            top_recs = sorted(
                candidates.values(), key=lambda x: x['score'], reverse=True
            )[:max_recs]

            self.recommendations[customer_id] = [
                {
                    'StockCode': r['product'],
                    'Description': self.product_lookup.get(r['product'], r['product']),
                    'lift': round(r['lift'], 3),
                    'confidence': round(r['confidence'], 3),
                    'score': round(r['score'], 3)
                }
                for r in top_recs
            ]
            success += 1

        total = len(all_customers)
        print(f"\n✓ Recommendations generated")
        print(f"  Customers with recommendations : {success:,} ({success/total*100:.1f}%)")
        print(f"  Customers with no history      : {no_history:,}")
        print(f"  Customers with no rule matches : {no_rules:,}")

        return self

    # =========================================================================
    # STEP 5 — FORMAT OUTPUT & MERGE
    # =========================================================================

    def get_recommendations_dataframe(self) -> pd.DataFrame:
        """
        Convert the recommendations dict into a flat DataFrame for merging.

        Returns:
            DataFrame with columns: CustomerID, Recommended_Products
            where Recommended_Products is a pipe-separated string of
            product descriptions (e.g. "ITEM A | ITEM B | ITEM C")
        """
        rows = []
        for customer_id, recs in self.recommendations.items():
            if recs:
                rec_str = ' | '.join([r['Description'] for r in recs])
            else:
                rec_str = 'No recommendations available'
            rows.append({
                'CustomerID': customer_id,
                'Recommended_Products': rec_str
            })
        return pd.DataFrame(rows)

    def merge_with_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge recommendations into the main customer intelligence results table.

        Args:
            results_df: The enriched customer table from the pipeline.

        Returns:
            results_df with a new 'Recommended_Products' column appended.
        """
        rec_df = self.get_recommendations_dataframe()
        merged = results_df.merge(rec_df, on='CustomerID', how='left')
        merged['Recommended_Products'] = merged['Recommended_Products'].fillna(
            'No recommendations available'
        )
        print(f"\n✓ Recommendations merged into results table")
        print(f"  Customers with recs : "
              f"{(merged['Recommended_Products'] != 'No recommendations available').sum():,}")
        return merged

    # =========================================================================
    # STEP 6 — INSIGHTS & DISPLAY
    # =========================================================================

    def print_sample_recommendations(self, n: int = 5):
        """
        Print sample recommendations for n customers for sanity-checking.

        Args:
            n: Number of sample customers to display.
        """
        print(f"\n{SUBHEADER_LINE}")
        print(f"SAMPLE RECOMMENDATIONS (top {n} customers)")
        print(SUBHEADER_LINE)

        sample_customers = [
            cid for cid, recs in self.recommendations.items() if recs
        ][:n]

        for customer_id in sample_customers:
            recs = self.recommendations[customer_id]

            # Get segment info if available
            seg_label = ''
            if self.customer_segments is not None:
                seg_row = self.customer_segments[
                    self.customer_segments['CustomerID'] == customer_id
                ]
                if len(seg_row):
                    seg_label = f"[{seg_row['Segment'].values[0]}]"

            print(f"\n  Customer {customer_id} {seg_label}")
            for i, r in enumerate(recs, 1):
                print(f"    {i}. {r['Description'][:50]:<52} "
                      f"(lift={r['lift']:.2f}, conf={r['confidence']:.2f})")

    def get_top_recommended_products(self, top_n: int = 10) -> pd.DataFrame:
        """
        Return the most frequently recommended products across all customers.

        Useful for inventory planning and campaign prioritization.

        Args:
            top_n: Number of top products to return.

        Returns:
            DataFrame with columns: StockCode, Description, Times_Recommended
        """
        product_counts = defaultdict(int)
        for recs in self.recommendations.values():
            for r in recs:
                product_counts[r['StockCode']] += 1

        rows = [
            {
                'StockCode': code,
                'Description': self.product_lookup.get(code, code),
                'Times_Recommended': count
            }
            for code, count in product_counts.items()
        ]

        df = (
            pd.DataFrame(rows)
            .sort_values('Times_Recommended', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        print(f"\n{SUBHEADER_LINE}")
        print(f"TOP {top_n} MOST RECOMMENDED PRODUCTS")
        print(SUBHEADER_LINE)
        print(df.to_string(index=False))

        return df