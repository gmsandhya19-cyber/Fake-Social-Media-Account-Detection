import io
import unittest
from unittest.mock import patch
from html.parser import HTMLParser
from app import ModelUnavailable
import json
import pandas as pd
from app import app, FEATURES, KEYS, validate, predictor, BASE_DIR

class ApplicationTests(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.sample = pd.read_csv(BASE_DIR / 'data.csv').loc[:4, FEATURES]

    def test_pages(self):
        for path in ['/', '/about', '/view', '/model', '/dashboard', '/prediction', '/batch', '/batch/sample']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_prediction_preserves_features(self):
        values = [1, .27, 2, .1, 1, 53, 1, 1, 32, 1000, 955]
        frame = validate(pd.DataFrame([values], columns=FEATURES))
        self.assertEqual(frame.iloc[0].tolist(), values)
        response = self.client.post('/prediction', data=dict(zip(KEYS, values)))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'model confidence', response.data)

    def test_invalid_inputs(self):
        for column,value in [('profile pic',2), ('nums/length username',1.2), ('#posts',-1), ('#followers',1.5), ('private',float('nan')), ('#follows',float('inf'))]:
            with self.subTest(column=column):
                frame = self.sample.copy().astype(float)
                frame.loc[0,column] = value
                with self.assertRaises(ValueError):
                    validate(frame)
        self.assertEqual(self.client.post('/prediction', data={}).status_code, 400)

    def test_batch_predictions_and_extra_columns(self):
        frame = self.sample.copy()
        frame['account_id'] = ['a','b','c','d','e']
        response = self.client.post('/batch', data={'data':(io.BytesIO(frame.to_csv(index=False).encode()),'accounts.csv')})
        self.assertEqual(response.status_code,200)
        output = pd.read_csv(io.BytesIO(response.data))
        expected = ['Likely fake' if v else 'Likely genuine' for v in predictor().predict(self.sample)]
        self.assertEqual(output.prediction.tolist(),expected)
        self.assertEqual(output.account_id.tolist(),frame.account_id.tolist())
        self.assertTrue(output.confidence_percent.between(0,100).all())
        self.assertIn('attachment',response.headers['Content-Disposition'])

    def test_bad_uploads(self):
        for content in [b'wrong,column\n1,2', b'', self.sample.assign(**{'#posts':-1}).to_csv(index=False).encode()]:
            response = self.client.post('/batch',data={'data':(io.BytesIO(content),'accounts.csv')})
            self.assertEqual(response.status_code,400)
        with patch.dict(app.config, MAX_CONTENT_LENGTH=1024):
            response = self.client.post('/batch',data={'data':(io.BytesIO(b'x'*1025),'accounts.csv')})
            self.assertEqual(response.status_code,413)
            response.close()

    def test_duplicate_csv_header_rejected(self):
        csv = self.sample.to_csv(index=False)
        lines = csv.splitlines()
        content = '\n'.join([lines[0]+',profile pic']+[line+',1' for line in lines[1:]])
        response = self.client.post('/batch', data={'data':(io.BytesIO(content.encode()),'accounts.csv')})
        self.assertEqual(response.status_code,400)
        self.assertIn(b'column names must be unique',response.data)

    def test_large_numbers_rejected(self):
        row = self.sample.iloc[0].astype(float).tolist()
        row[9] = 1e100
        response = self.client.post('/prediction',data=dict(zip(KEYS,row)))
        self.assertEqual(response.status_code,400)
        self.assertIn(b'too large',response.data)

    def test_row_limit_and_empty_dataset(self):
        with self.assertRaises(ValueError):
            validate(self.sample.iloc[:0])
        with self.assertRaises(ValueError):
            validate(pd.concat([self.sample]*2001,ignore_index=True))

    def test_missing_model_has_helpful_error(self):
        with patch('app.predictor',side_effect=ModelUnavailable('Run train_model.py to rebuild the model.')):
            response = self.client.post('/prediction',data=dict(zip(KEYS,self.sample.iloc[0].tolist())))
        self.assertEqual(response.status_code,503)
        self.assertIn(b'train_model.py',response.data)

    def test_missing_metrics_has_helpful_error(self):
        with patch('app.Path.read_text',side_effect=FileNotFoundError()):
            response = self.client.get('/dashboard')
        self.assertEqual(response.status_code,503)

    def test_internal_links_and_assets(self):
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.targets = set()
            def handle_starttag(self,tag,attrs):
                for key,value in attrs:
                    if key in ('href','src') and value and value.startswith('/'):
                        self.targets.add(value)
        for path in ['/', '/about', '/view', '/prediction', '/dashboard', '/batch']:
            parser = Links()
            parser.feed(self.client.get(path).get_data(as_text=True))
            for target in parser.targets:
                with self.subTest(page=path,target=target):
                    response = self.client.get(target)
                    self.assertEqual(response.status_code,200)
                    response.close()
        self.assertEqual(self.client.get('/does-not-exist').status_code,404)

    def test_saved_models_and_metrics_agree(self):
        import pickle
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        data = pd.read_csv(BASE_DIR / 'data.csv').drop_duplicates()
        _, x, _, y = train_test_split(validate(data),data['fake'],test_size=.3,stratify=data['fake'],random_state=42)
        metrics = json.loads((BASE_DIR / 'model_metrics.json').read_text())
        for name,filename in [('Random Forest','Random_forest.sav'),('Decision Tree','Decision_tree.sav'),('Logistic Regression','Logistic_Regression.sav')]:
            with (BASE_DIR / filename).open('rb') as handle:
                model = pickle.load(handle)
            self.assertAlmostEqual(accuracy_score(y,model.predict(x)),metrics['models'][name]['accuracy'])

if __name__ == '__main__':
    unittest.main()
