echo "Sending score request to containerized model endpoint..."
echo "***"
echo "***"
curl \
'https://demo.dominodatalab.com:443/models/60118d880de5e1255a29096f/latest/model' \
-H 'Content-Type: application/json' \
-d '{"data": {
    "dropperc": 0.1,
    "mins": 100,
    "consecmonths": 6,
    "age": 45,
    "income": 200}}' \
-u Gl1Dvu7cHdUzMT3eg4jqzitVwoZZmnRy9R2cv9u2lRmtK4tHJQEHC0VYA42m3PmJ:Gl1Dvu7cHdUzMT3eg4jqzitVwoZZmnRy9R2cv9u2lRmtK4tHJQEHC0VYA42m3PmJ


echo "DONE."