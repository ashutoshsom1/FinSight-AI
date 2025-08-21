# Sample Financial Reports

This directory contains sample financial reports for testing the FinSight AI system.

## Directory Structure

```
sample_reports/
├── castrol/
│   ├── castrol_annual_report_2023.pdf
│   ├── castrol_annual_report_2022.pdf
│   └── castrol_financial_data_2023.csv
├── veedol/
│   ├── veedol_annual_report_2023.pdf
│   ├── veedol_annual_report_2022.pdf
│   └── veedol_financial_data_2023.csv
└── valvoline/
    ├── valvoline_annual_report_2023.pdf
    ├── valvoline_annual_report_2022.pdf
    └── valvoline_financial_data_2023.csv
```

## How to Add Reports

1. **Download Annual Reports**:
   - **Castrol**: Download from BP's investor relations page
   - **Veedol**: Download from BSE/NSE or company website
   - **Valvoline**: Download from NYSE or company investor relations

2. **File Naming Convention**:
   - Format: `{company}_annual_report_{year}.pdf`
   - Example: `castrol_annual_report_2023.pdf`

3. **Supported Formats**:
   - PDF files (annual reports)
   - CSV/Excel files (financial data tables)

## Sample Data Sources

### Castrol (BP Subsidiary)
- **Website**: https://www.bp.com/en/global/corporate/investors.html
- **Reports**: Annual reports available in PDF format
- **Currency**: GBP
- **Fiscal Year**: December

### Veedol (Tide Water Oil Company)
- **Website**: https://www.veedol.com/investor-relations
- **Exchange**: BSE/NSE (India)
- **Currency**: INR
- **Fiscal Year**: March

### Valvoline
- **Website**: https://investors.valvoline.com
- **Exchange**: NYSE
- **Currency**: USD
- **Fiscal Year**: September

## Testing the System

Once you've added sample reports:

1. **Ingest documents**:
   ```bash
   python scripts/ingest_reports.py --directory ./data/sample_reports
   ```

2. **Test the system**:
   ```bash
   python scripts/test_system.py
   ```

3. **Start the API**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **Test queries**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "What was Castrol'\''s revenue in 2023?"}'
   ```

## Sample Queries to Test

- "What was Castrol's revenue growth in 2023?"
- "Compare operating margins between Valvoline and Castrol"
- "What are the main risk factors for Veedol?"
- "How did Valvoline's debt-to-equity ratio change over the last 3 years?"
- "Summarize Castrol's financial performance in 2023"

## Notes

- Ensure you have set your OpenAI API key in the `.env` file
- The system will automatically detect company and year from file names
- For best results, use the most recent annual reports (2022-2024)
- Financial data in CSV format should include year columns for proper temporal analysis
