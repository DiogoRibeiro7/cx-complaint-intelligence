# CFPB Complaint Intelligence Executive Summary

## Overview

- Total complaints analysed: **1,000**.
- High-risk complaints: **136** (13.6%).
- Medium or high-risk complaints: **679**.

The risk score is an interpretable prioritisation signal. It combines response timeliness, product and issue risk, urgency language, monetary language, narrative length, and CFPB tags.

## Top products

| product                                                 |   complaints |   average_risk_score |   high_risk_share |
|:--------------------------------------------------------|-------------:|---------------------:|------------------:|
| Credit reporting or other personal consumer reports     |          793 |                 3.58 |              14.6 |
| Debt collection                                         |           77 |                 3.81 |              15.6 |
| Credit card                                             |           43 |                 2.28 |               2.3 |
| Checking or savings account                             |           40 |                 1.99 |               0   |
| Mortgage                                                |           13 |                 3.63 |              15.4 |
| Student loan                                            |           11 |                 1.91 |               0   |
| Payday loan, title loan, personal loan, or advance loan |            9 |                 4.11 |              33.3 |
| Money transfer, virtual currency, or money service      |            7 |                 3.43 |              28.6 |
| Vehicle loan or lease                                   |            4 |                 1.81 |               0   |
| Prepaid card                                            |            2 |                 2.38 |               0   |

## Top issues

| issue                                                           |   complaints |   average_risk_score |   high_risk_share |
|:----------------------------------------------------------------|-------------:|---------------------:|------------------:|
| Incorrect information on your report                            |          369 |                 4.46 |              29   |
| Improper use of your report                                     |          227 |                 2.8  |               3.1 |
| Problem with a company's investigation into an existing problem |          185 |                 2.73 |               1.1 |
| Attempts to collect debt not owed                               |           39 |                 4.49 |              20.5 |
| Managing an account                                             |           23 |                 2.13 |               0   |
| Written notification about debt                                 |           21 |                 2.54 |               0   |
| Other features, terms, or problems                              |           11 |                 1.91 |               0   |
| Problem with a purchase shown on your statement                 |            9 |                 2.97 |               0   |
| Closing an account                                              |            9 |                 1.5  |               0   |
| Problem with fraud alerts or security freezes                   |            8 |                 4.38 |              12.5 |

## Top companies

| company                                |   complaints |   average_risk_score |   high_risk_share |
|:---------------------------------------|-------------:|---------------------:|------------------:|
| TRANSUNION INTERMEDIATE HOLDINGS, INC. |          285 |                 3.5  |              12.6 |
| EQUIFAX, INC.                          |          253 |                 3.56 |              14.2 |
| Experian Information Solutions Inc.    |          230 |                 3.6  |              14.3 |
| CITIBANK, N.A.                         |           17 |                 2.66 |               0   |
| WELLS FARGO & COMPANY                  |           17 |                 2.38 |               0   |
| JPMORGAN CHASE & CO.                   |           15 |                 2    |               0   |
| BANK OF AMERICA, NATIONAL ASSOCIATION  |            9 |                 3.56 |              22.2 |
| CAPITAL ONE FINANCIAL CORPORATION      |            8 |                 2.28 |              12.5 |
| GOLDMAN SACHS BANK USA                 |            7 |                 2.29 |               0   |
| Resurgent Capital Services L.P.        |            6 |                 4.38 |              16.7 |

## Topic distribution

|   topic_id |   complaints |   average_risk_score |   high_risk_share |
|-----------:|-------------:|---------------------:|------------------:|
|          4 |          410 |                 3.4  |               7.8 |
|          0 |          329 |                 3.66 |              21.6 |
|          3 |           66 |                 3.88 |              36.4 |
|          1 |           63 |                 2.91 |               3.2 |
|          7 |           52 |                 3.56 |              11.5 |
|          2 |           39 |                 3.16 |               0   |
|          6 |           21 |                 2.82 |               0   |
|          5 |           20 |                 2.48 |               5   |

## Topic labels from top terms

- **Topic 0**: redacted, redacted redacted, money, account, balance, redacted balance, money redacted, year
- **Topic 1**: consumer, 15, section, states, 1681, consumer reporting, reporting agency, agency
- **Topic 2**: accounts, fcra 605b, significance eliminating, eliminating erroneous, eliminating, significance, noticed discrepancies, ve noticed
- **Topic 3**: payment, late, payments, creditor, late payment, error, usc, payment late
- **Topic 4**: credit, report, credit report, information, accounts, account, remove, request
- **Topic 5**: payments, perplexing discover, perplexing, updated records, maintained time, reflect timely, report seeking, payments perplexing
- **Topic 6**: letters, party involved, really sure, falsely reporting, happened mailed, far gotten, gotten response, involved review
- **Topic 7**: consumer reports, consent, consumer, reports, 15 code, code, docs verification, docs

## Recommended next actions

1. Review high-risk complaints first and validate whether the score aligns with analyst judgement.
2. Use topic groups to identify recurring process failures, not only isolated customer issues.
3. Compare companies only after normalising by exposure, customer base, or complaint volume context.
4. Track topic and issue trends over time to detect emerging customer harm earlier.
5. Convert validated high-risk labels into a supervised model once enough reviewed examples exist.

## Limitations

- The score is heuristic and should not be interpreted as a legal or regulatory judgement.
- CFPB narratives are available only when consumers consent to publication.
- Company comparisons require external normalisation before strong conclusions are made.
