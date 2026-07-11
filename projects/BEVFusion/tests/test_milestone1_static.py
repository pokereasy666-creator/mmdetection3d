import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / 'bevfusion'


def _class_node(path, class_name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name)


def _method_node(class_node, method_name):
    return next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name)


def test_all_off_path_stays_on_original_extract_feat():
    detector = _class_node(PACKAGE_ROOT / 'bevfusion.py', 'BEVFusion')
    extract_feat = _method_node(detector, 'extract_feat')
    called_attributes = {
        node.func.attr
        for node in ast.walk(extract_feat)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert 'extract_feature_bundle' not in called_attributes
    assert 'instance_refiner' not in {
        node.attr
        for node in ast.walk(extract_feat) if isinstance(node, ast.Attribute)
    }


def test_default_image_path_does_not_retain_raw_pyramid():
    detector = _class_node(PACKAGE_ROOT / 'bevfusion.py', 'BEVFusion')
    extract_img_feat = _method_node(detector, 'extract_img_feat')
    raw_assignment_index = next(
        index for index, node in enumerate(extract_img_feat.body)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'raw_img_feats'
            for target in node.targets))
    baseline_branch_index = next(
        index for index, node in enumerate(extract_img_feat.body)
        if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not))
    baseline_branch = extract_img_feat.body[baseline_branch_index]
    assert baseline_branch_index < raw_assignment_index
    assert any(
        isinstance(node, ast.Return) for node in ast.walk(baseline_branch))


def test_tensor_mode_keeps_source_commit_behavior():
    detector = _class_node(PACKAGE_ROOT / 'bevfusion.py', 'BEVFusion')
    tensor_forward = _method_node(detector, '_forward')
    assert len(tensor_forward.body) == 2
    assert isinstance(tensor_forward.body[-1], ast.Pass)


def test_data_structures_are_not_modules_and_do_not_clone_tensors():
    path = PACKAGE_ROOT / 'structures.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    class_names = {
        'SensorMeta', 'BEVGeometry', 'RefinementFeatureSources',
        'FeatureBundle', 'ProposalPack'
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            assert not node.bases
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'clone' for node in ast.walk(tree))


def test_proposal_pack_has_one_query_major_contract():
    structures = ast.parse(
        (PACKAGE_ROOT / 'structures.py').read_text(encoding='utf-8'))
    proposal_pack = next(
        node for node in structures.body
        if isinstance(node, ast.ClassDef) and node.name == 'ProposalPack')
    fields = tuple(
        node.target.id for node in proposal_pack.body if
        isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name))
    assert fields == ('query_feat_pre', 'query_feat_post', 'ref_xy', 'scores',
                      'class_scores', 'labels', 'indices', 'dense_heatmap',
                      'source_type')


def test_milestone1_has_no_depth_or_fuser_specific_route():
    detector_source = (PACKAGE_ROOT /
                       'bevfusion.py').read_text(encoding='utf-8')
    lidar_source = (PACKAGE_ROOT / 'insfusion' /
                    'lidar_proposal.py').read_text(encoding='utf-8')
    combined = detector_source + lidar_source
    assert 'depth_supervisor' not in combined
    assert 'loss_depth' not in combined
    assert 'DGFFuser' not in combined
    assert 'ConvFuser' not in combined
    assert 'DummyAlternativeFuser' not in combined


def test_refinement_sources_exclude_non_kv_bundle_fields():
    structures = ast.parse(
        (PACKAGE_ROOT / 'structures.py').read_text(encoding='utf-8'))
    source_class = next(
        node for node in structures.body if isinstance(node, ast.ClassDef)
        and node.name == 'RefinementFeatureSources')
    fields = {
        node.target.id
        for node in source_class.body if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    assert fields == {
        'raw_img_feats', 'lidar_bev', 'fused_bev', 'sensor_meta',
        'bev_geometry'
    }


def test_transfusion_public_proposal_utility_accepts_head_feature():
    head = _class_node(PACKAGE_ROOT / 'transfusion_head.py', 'TransFusionHead')
    utility = _method_node(head, 'extract_proposals')
    called_attributes = {
        node.func.attr
        for node in ast.walk(utility)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert 'shared_conv' in called_attributes
    assert '_extract_proposals_from_shared_feature' in called_attributes


def test_freeze_contract_is_declared_but_not_executed():
    structures = ast.parse(
        (PACKAGE_ROOT / 'structures.py').read_text(encoding='utf-8'))
    assignment = next(
        node for node in structures.body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == 'BASELINE_FREEZE_MODULE_NAMES'
            for target in node.targets))
    module_names = tuple(value.value for value in assignment.value.elts)
    assert 'fusion_layer' in module_names
    assert 'depth_supervisor' in module_names
    assert 'instance_refiner' not in module_names

    detector = _class_node(PACKAGE_ROOT / 'bevfusion.py', 'BEVFusion')
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == 'train'
        for node in detector.body)
