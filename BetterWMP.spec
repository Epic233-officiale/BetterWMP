# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata
import sys

block_cipher = None

# Collect everything from scipy and pyloudnorm
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
pyln_datas, pyln_binaries, pyln_hiddenimports = collect_all('pyloudnorm')

# Force all submodules
all_scipy_submodules = collect_submodules('scipy')
all_pyln_submodules = collect_submodules('pyloudnorm')
all_mutagen_submodules = collect_submodules('mutagen')

a = Analysis(
    ['betterwmp sourcecode.py'],
    pathex=[],
    binaries=scipy_binaries + pyln_binaries,
    datas=scipy_datas + pyln_datas + copy_metadata('scipy') + copy_metadata('pyloudnorm'),
    hiddenimports=[
        # Force all scipy submodules
        *all_scipy_submodules,
        *all_pyln_submodules,
        *all_mutagen_submodules,
        # Core dependencies
        'numpy',
        'sounddevice',
        'soundfile',
        'pydub',
        'mutagen',
        'PIL',
        'tkinterdnd2',
        'pynput',
        'keyboard',
        # Additional scipy internals that might be lazy-loaded
        'scipy.signal',
        'scipy.signal.windows',
        'scipy.fft',
        'scipy.fft._pocketfft',
        'scipy.special',
        'scipy.special._ufuncs',
        'scipy.special._ufuncs_cxx',
        'scipy.sparse',
        'scipy.sparse.csgraph',
        'scipy.linalg',
        'scipy.integrate',
        'scipy.io',
        'scipy.io.wavfile',
        'scipy._lib',
        'scipy._lib.messagestream',
        'scipy._lib._ccallback',
        'scipy.stats',
        'scipy.stats._stats_py',
        'scipy.stats.distributions',
        'scipy.stats._distn_infrastructure',
        'scipy.stats._continuous_distns',
        'scipy.stats._discrete_distns',
        # Pyloudnorm internals
        'pyloudnorm.normalize',
        'pyloudnorm.meter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Patch scipy stats module before building
for d in a.datas:
    if d[0] == 'scipy/stats/_distn_infrastructure.py':
        a.datas.remove(d)
        break

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BetterWMP',
    icon=r'F:\Workspace\Py\BetterWMP\icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BetterWMP'
)