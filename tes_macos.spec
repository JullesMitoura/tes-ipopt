# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

cyipopt_datas, cyipopt_binaries, cyipopt_hidden = collect_all('cyipopt')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=cyipopt_binaries,
    datas=[
        ('app', 'app'),
        ('ico.ico', '.'),
    ] + cyipopt_datas,
    hiddenimports=[
        'cyipopt',
        'pyomo.environ',
        'pyomo.core',
        'pyomo.solvers',
        'scipy.optimize',
        'scipy.sparse',
        'scipy.sparse.linalg',
    ] + cyipopt_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TeS_App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='ico.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TeS_App',
)

app = BUNDLE(
    coll,
    name='TeS_App.app',
    icon='ico.ico',
    bundle_identifier='com.tes.thermoequilibrium',
    info_plist={
        'CFBundleDisplayName': 'TeS App',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,
    },
)
