from monitoring.feature_extractor import FeatureExtractor


def test_feature_extractor_transformations():
    logs = ["error on line 1", "warning: disk full"]
    extractor = FeatureExtractor()
    matrix = extractor.fit_transform(logs)
    assert matrix.shape[0] == 2

    new_logs = ["error on line 2"]
    new_matrix = extractor.transform(new_logs)
    assert new_matrix.shape[0] == 1
