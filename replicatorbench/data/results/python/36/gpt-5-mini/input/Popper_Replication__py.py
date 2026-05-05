import os
import sys
import pandas as pd
import numpy as np

OUTDIR = '/app/data/replication_results'
os.makedirs(OUTDIR, exist_ok=True)

# Use workspace-mounted path at runtime (container runs with /workspace as CWD)
DATA_PATH = os.path.join(os.getcwd(), 'replication_data', 'Popper Replication Data Files', 'Popper_Data for CFA and SEM.csv')

def main():
    try:
        import semopy
    except Exception as e:
        print('Missing dependency semopy or import failed:', e)
        print('Please ensure semopy is installed in the environment.')
        sys.exit(2)

    print('Loading data from', DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print('Data shape:', df.shape)
    # Save a head for inspection
    df.head().to_csv(os.path.join(OUTDIR, 'data_head.csv'), index=False)

    # Define measurement model and structural Model 1 (primary mediation)    # Define measurement model and structural Model 1 (primary mediation)
    measurement = (
        'ATTACH =~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3\n'
        'ANXIETY =~ STAI_Par1 + STAI_Par2 + STAI_Par3\n'
        'OPEN =~ Open_Par1 + Open_Par2 + Open_Par3\n'
        'LEAD =~ Lead_Par1 + Lead_Par2 + Lead_Par3'
    )

    model1 = measurement + '\n' + (
        'ANXIETY ~ a*ATTACH\n'
        'OPEN ~ c*ATTACH\n'
        'LEAD ~ b*ANXIETY + d*OPEN\n'
    )

    # Note: semopy may not support defined parameters in all versions with ':=' syntax.
    # We'll compute indirect effects after fitting by multiplying path coefficients (ab = a*b; cd = c*d).

    print('Fitting SEM Model 1...')
    try:
        mod = semopy.Model(model1)
        res = mod.fit(df)
    except Exception as e:
        print('SEM fit failed:', e)
        sys.exit(3)

    # Parameter table    # Parameter table
    a = b = c = d = None
    try:
        params = mod.inspect()
        if params is not None:
            params.to_csv(os.path.join(OUTDIR, 'model1_parameters.csv'), index=False)
    except Exception:
        params = None
    # Try alternative attribute
    if params is None:
        try:
            df_params = getattr(mod, 'parameters_', None)
            if df_params is not None:
                df_params.to_csv(os.path.join(OUTDIR, 'model1_parameters.csv'), index=False)
                # Attempt to extract path coefficients by name
                # semopy parameter tables often have 'name' or 'param' and 'value'
                if 'name' in df_params.columns and 'value' in df_params.columns:
                    for _, row in df_params.iterrows():
                        name = str(row.get('name', '')).strip()
                        val = float(row.get('value', np.nan))
                        if name.endswith('~ATTACH') and name.startswith('ANXIETY'):
                            a = val
                        if name.endswith('~ATTACH') and name.startswith('OPEN'):
                            c = val
                        if name.endswith('~ANXIETY') and name.startswith('LEAD'):
                            b = val
                        if name.endswith('~OPEN') and name.startswith('LEAD'):
                            d = val
        except Exception:
            pass
    else:
        # Try to extract from params DataFrame
        try:
            for _, row in params.iterrows():
                lhs = str(row.get('lval', '')) if 'lval' in params.columns else str(row.get('lhs', ''))
                rhs = str(row.get('rval', '')) if 'rval' in params.columns else str(row.get('rhs', ''))
                est = float(row.get('Estimate', row.get('est', row.get('value', np.nan))))
                # Match patterns
                if lhs == 'ANXIETY' and rhs == 'ATTACH':
                    a = est
                if lhs == 'OPEN' and rhs == 'ATTACH':
                    c = est
                if lhs == 'LEAD' and rhs == 'ANXIETY':
                    b = est
                if lhs == 'LEAD' and rhs == 'OPEN':
                    d = est
        except Exception:
            pass

    # Compute indirects if available
    indirects = {}
    if a is not None and b is not None:
        indirects['ab'] = a * b
    if c is not None and d is not None:
        indirects['cd'] = c * d
    pd.DataFrame([indirects]).to_csv(os.path.join(OUTDIR, 'model1_indirects.csv'), index=False)
    # Fit stats
    try:
        stats = semopy.calc_stats(mod)
        with open(os.path.join(OUTDIR, 'model1_fit_stats.txt'), 'w') as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
    except Exception as e:
        print('Failed to compute fit stats:', e)

    print('Model 1 fitting completed. Outputs saved to', OUTDIR)

if __name__ == '__main__':
    main()
