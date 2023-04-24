SOURCE=source
all: libraries run

libraries:
	python3 -m pip install -r requirements.txt

run:
	python3 $(SOURCE)/age_prediction.py

