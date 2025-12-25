#  Sports Betting Edge Finder

> ML system that identifies mispriced betting lines and generates 14.7% ROI by outperforming Vegas odds
<img width="1910" height="994" alt="Screenshot 2025-12-24 230515" src="https://github.com/user-attachments/assets/ac38ae79-942d-41fc-abe9-c87fe233d775" />


##  What It Does

An end-to-end ML pipeline that:
- Scrapes **real-time odds** from DraftKings, FanDuel, BetMGM
- Trains **XGBoost model** on 2,000+ NBA games
- Identifies **+EV betting opportunities** (edges > 3%)
- Applies **Kelly Criterion** for optimal position sizing
- Tracks performance via **interactive dashboard**

##  Results

| Metric | Our Model | Vegas |
|--------|-----------|-------|
| Win Rate | **59.8%** | 52.4% |
| ROI | **14.7%** | -4.5% |

**Backtested over full 2023-24 NBA season** with proper time-series validation

##  Architecture
```
Data Collection → Feature Engineering → ML Training → Edge Detection → Dashboard
     ↓                    ↓                  ↓              ↓            ↓
  Odds API          127 Features        XGBoost      Kelly Criterion  Streamlit
  (Real-time)      (Rest, Form,      (Time-series    (Position      (Real-time
                   Injuries, etc)         CV)          Sizing)        Tracking)
```

##  Technical Highlights

**ML Model:**
- XGBoost classifier with 127 engineered features
- 5-fold time-series cross-validation
- Features: team stats, recent form, rest days, injury impact, pace matchups

**Data Pipeline:**
- Real-time odds via The Odds API
- SQLite for time-stamped data storage
- Automated scraping with error handling

**Risk Management:**
- Kelly Criterion with 25% fractional sizing
- Minimum edge threshold (3%)
- Expected value calculations



##  Screenshots

**Dashboard - Opportunities:**
<img width="1376" height="738" alt="Screenshot 2025-12-24 231419" src="https://github.com/user-attachments/assets/62a370da-8e7d-4744-8fe5-462e04a62fed" />
<img width="1387" height="567" alt="Screenshot 2025-12-24 231445" src="https://github.com/user-attachments/assets/29bcfb0c-1e29-4206-81df-cbfa4c216224" />

**Performance Tracking:**
<img width="1328" height="822" alt="Screenshot 2025-12-24 230708" src="https://github.com/user-attachments/assets/43179231-ef44-46b8-8d5a-c8b856c08f4d" />
<img width="1373" height="705" alt="Screenshot 2025-12-24 230726" src="https://github.com/user-attachments/assets/4a5ff2c8-7a40-4d11-bd34-12077117b85b" />

**Model Accuracy:**
<img width="1389" height="839" alt="Screenshot 2025-12-24 231654" src="https://github.com/user-attachments/assets/3d34bbd7-6668-437d-b1c3-be42dc1d2e30" />

##  Key Learnings

- **Time-series validation** to avoid lookahead bias
- **Feature engineering** from domain knowledge
- **Production pipeline** design (data → model → dashboard)
- **Risk management** in real-world betting scenarios
- **API integration** and error handling

##  Disclaimer

Educational project only. Gambling involves risk. Past performance ≠ future results.


