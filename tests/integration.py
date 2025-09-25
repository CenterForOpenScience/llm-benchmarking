import os, pytest, json
import info_extractor.extractor as extractor
import generator.design.easy as gen_easy
import logging

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("API_KEY"), reason="No API_KEY set for real GPT test")
def test_extractor_stage1_real(tmp_path):
    result = extractor.run_stage_1(study_path="case_studies/case_study_1", difficulty="easy")
    assert isinstance(result, dict)
    # Optionally validate against JSON schema keys
    assert "original_study" in result

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("API_KEY"), reason="No API_KEY set for real GPT test")
def test_generator_design_easy_real(tmp_path):
    logger = logging.getLogger("integration")
    prereg = gen_easy.run_design_easy(
        study_path="case_studies/case_study_1",
        templates_dir="./templates",
        show_prompt=False,
        logger=logger
    )
    assert isinstance(prereg, dict)
    # schema keys expected
    assert "original_study" in prereg
    assert "replication" in prereg

