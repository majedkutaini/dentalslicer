from gui import InteractiveRotationViewer

# v: Model name
# xx: Tooth number (FDI Schema)
v = "M16"
xx = "34"

if __name__ == '__main__':
    model_file = f'Zähne/{v}/{xx}/{xx}_umschlagpunkt_neu.stl' # Path to the intraoral digital impression including surrounding structures. (STL)
    prep_file = f'Zähne/{v}/{xx}/{xx}_präpgrenze_d.stl'  # Path to the intraoral digital impression (STL)
    prep_k_file = f'Zähne/{v}/{xx}/{xx}_präpgrenze_k.stl' # Path to the conventional scanned preparation (STL)
    
    print(f"Starting viewer for Tooth {xx} (Model {v})")
    print(f"Loading files:\n- {model_file}\n- {prep_file}\n- {prep_k_file}")
    
    viewer = InteractiveRotationViewer(
        model_file, prep_file, prep_k_file, 
        v=v, xx=xx,
        num_angles=360, 
        slice_resolution=1200, 
        batch_size=60
    )
    viewer.run()

