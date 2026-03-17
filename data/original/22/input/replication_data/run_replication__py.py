import importlib.util
import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(path):
    spec = importlib.util.spec_from_file_location("_tmp_module", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tmp_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    # Ensure data is available at /app/data by copying from local replication_data if needed
    src = os.path.join(HERE, "AlTammemi_Survey_deidentify.csv")
    dest_dir = "/app/data"
    try:
        if os.path.exists(src) and os.path.isdir(dest_dir) and not os.path.exists(os.path.join(dest_dir, "AlTammemi_Survey_deidentify.csv")):
            print("Copying dataset to /app/data ...")
            shutil.copy2(src, os.path.join(dest_dir, "AlTammemi_Survey_deidentify.csv"))
    except Exception as e:
        print(f"Warning: could not copy dataset into /app/data automatically: {e}")

    focal = os.path.join(HERE, "analysis_dl_outcome__py.py")
    alt = os.path.join(HERE, "analysis__py.py")

    print("Running focal replication: Ordered logistic with motivation as outcome...")
    mod_focal = load_module(focal)
    if hasattr(mod_focal, "main") and callable(mod_focal.main):
        mod_focal.main()
    elif hasattr(mod_focal, "load_and_prepare") and callable(mod_focal.load_and_prepare):
        mod_focal.load_and_prepare()
    else:
        raise RuntimeError("Focal module has no callable entrypoint (main or load_and_prepare)")

    print("Running alternative specification: MNLogit with distress as outcome...")
    mod_alt = load_module(alt)
    if hasattr(mod_alt, "main") and callable(mod_alt.main):
        mod_alt.main()
    elif hasattr(mod_alt, "run") and callable(mod_alt.run):
        mod_alt.run()
    else:
        raise RuntimeError("Alternative module has no callable entrypoint (main or run)")


if __name__ == "__main__":
    main()
