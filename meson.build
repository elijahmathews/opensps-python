project('opensps', 'c', 'cython',
  version : '0.1.0.dev0',
  license : 'MIT',
  meson_version : '>=1.4',
  default_options : ['buildtype=release', 'b_ndebug=if-release'],
)

py = import('python').find_installation(pure : false)
numpy_dep = dependency('numpy')

# ------------------------------------------------------------------------------------------
# The core library (libfsps, later libopensps). Two supported sources:
#  (a) an installed library found through pkg-config (names in -Dcore_names, tried in order);
#  (b) otherwise the Meson subproject subprojects/fsps.wrap, built static into the extension
#      (no test programs, no data install). Force one with
#      -Csetup-args=--wrap-mode=nofallback (a) or --wrap-mode=forcefallback (b).
# ------------------------------------------------------------------------------------------
core_dep = dependency(get_option('core_names'),
  fallback : ['fsps', 'fsps_dep'],
  default_options : ['tests=false', 'install_data=false', 'default_library=static'],
)

ext_rpath = ''
if core_dep.type_name() == 'pkgconfig'
  # An installed shared library outside the default search path: find it at run time
  ext_rpath = core_dep.get_variable(pkgconfig : 'libdir')
else
  # The subproject's Fortran objects are linked into the extension
  add_languages('fortran', native : false)
endif

py.extension_module('_core',
  'src/opensps/_core.pyx',
  dependencies : [core_dep, numpy_dep],
  include_directories : include_directories('src/opensps'),
  install : true,
  subdir : 'opensps',
  install_rpath : ext_rpath,
  override_options : ['cython_language=c'],
)

py.install_sources(
  'src/opensps/__init__.py',
  'src/opensps/_config.py',
  'src/opensps/dust.py',
  'src/opensps/filters.py',
  'src/opensps/_capi.pxd',
  subdir : 'opensps',
)

py.install_sources(
  'src/opensps/compat/__init__.py',
  'src/opensps/compat/fsps.py',
  subdir : 'opensps/compat',
)
