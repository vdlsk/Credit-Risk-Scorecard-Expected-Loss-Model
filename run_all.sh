#!/usr/bin/env bash
# Full pipeline: data -> scorecard -> validation -> ECL & capital
set -e
python src/get_data.py
python src/prepare_data.py
python src/build_scorecard.py
python src/validate.py
python src/challenger.py
python src/ecl_capital.py
echo "Done. Launch the app with:  streamlit run app/app.py"
