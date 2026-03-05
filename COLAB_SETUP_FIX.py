"""
PASTE THIS AS A NEW CELL IN COLAB (before the 'from r2d_hope import' line).
Copies r2d_hope/ from Drive to /content and adds it to sys.path.

PREREQUISITE: Upload r2d_hope.zip to MyDrive first.
  - On your Mac: zip -r r2d_hope.zip r2d_hope/  (run in the project folder)
  - Upload the zip to Google Drive via drive.google.com
"""
import os, sys, shutil

DRIVE_ZIP = '/content/drive/MyDrive/r2d_hope.zip'
DEST_DIR  = '/content'

if not os.path.exists(os.path.join(DEST_DIR, 'r2d_hope')):
    if os.path.exists(DRIVE_ZIP):
        shutil.unpack_archive(DRIVE_ZIP, DEST_DIR)
        print(f'Unzipped r2d_hope from Drive → {DEST_DIR}/r2d_hope')
    else:
        raise FileNotFoundError(
            f'{DRIVE_ZIP} not found.\n'
            'Please upload r2d_hope.zip to the root of your Google Drive.\n'
            'On your Mac, run:  zip -r r2d_hope.zip r2d_hope/\n'
            'then upload via drive.google.com'
        )
else:
    print('r2d_hope/ already present at /content/r2d_hope')

if DEST_DIR not in sys.path:
    sys.path.insert(0, DEST_DIR)

from r2d_hope import R2DConfig, R2D_HOPE_MoRE
print('r2d_hope package loaded successfully.')
