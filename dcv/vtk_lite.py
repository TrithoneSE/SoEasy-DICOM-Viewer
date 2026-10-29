"""The VTK classes this application uses, imported module by module.

`import vtk` loads every VTK library (about 200 DLLs, 243 MB in a frozen build); importing only these modules lets
the build carry just what is needed. Code uses it as `from . import vtk_lite as vtk`, so `vtk.vtkActor` etc. keep
working. A class missing here raises AttributeError at once (the GUI tests import this module too).
"""

# object factories: OpenGL render window, GPU volume mapper, text rendering, default interactor style
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingFreeType  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
from vtkmodules.util.vtkConstants import VTK_FLOAT, VTK_SHORT, VTK_UNSIGNED_CHAR  # noqa: F401
from vtkmodules.vtkCommonCore import vtkFileOutputWindow, vtkOutputWindow, vtkPoints, vtkVersion  # noqa: F401
from vtkmodules.vtkCommonDataModel import (  # noqa: F401
    vtkCellArray, vtkImageData, vtkPiecewiseFunction, vtkPlanes, vtkPolyData,
)
from vtkmodules.vtkCommonMath import vtkMatrix4x4  # noqa: F401
from vtkmodules.vtkCommonTransforms import vtkTransform  # noqa: F401
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkTriangleFilter, vtkTubeFilter  # noqa: F401
from vtkmodules.vtkFiltersSources import vtkConeSource, vtkPolyLineSource, vtkSphereSource  # noqa: F401
from vtkmodules.vtkImagingCore import vtkImageCast, vtkImageReslice  # noqa: F401
from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth  # noqa: F401
from vtkmodules.vtkImagingMorphological import vtkImageConnectivityFilter  # noqa: F401
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleUser  # noqa: F401
from vtkmodules.vtkInteractionWidgets import (  # noqa: F401
    vtkBoxRepresentation, vtkBoxWidget2, vtkOrientationMarkerWidget,
)
from vtkmodules.vtkIOGeometry import vtkSTLWriter  # noqa: F401
from vtkmodules.vtkIOImage import vtkPNGWriter  # noqa: F401
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter  # noqa: F401
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor  # noqa: F401
from vtkmodules.vtkRenderingCore import (  # noqa: F401
    vtkActor, vtkActor2D, vtkBillboardTextActor3D, vtkCamera, vtkCameraInterpolator, vtkCellPicker,
    vtkColorTransferFunction, vtkCoordinate, vtkImageActor, vtkPolyDataMapper, vtkPolyDataMapper2D, vtkRenderer,
    vtkRenderWindow, vtkTextActor, vtkTextRenderer, vtkTransformInterpolator, vtkVolume, vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper, vtkVolumePicker  # noqa: F401

_FACTORIES = (vtkmodules.vtkInteractionStyle, vtkmodules.vtkRenderingFreeType, vtkmodules.vtkRenderingOpenGL2,
              vtkmodules.vtkRenderingVolumeOpenGL2)      # imported for their side effects (object factories)

__all__ = ["VTK_FLOAT", "VTK_SHORT", "VTK_UNSIGNED_CHAR", "vtkActor", "vtkActor2D", "vtkAnnotatedCubeActor",
           "vtkBillboardTextActor3D", "vtkBoxRepresentation", "vtkBoxWidget2", "vtkCamera", "vtkCameraInterpolator",
           "vtkCellArray", "vtkCellPicker", "vtkColorTransferFunction", "vtkConeSource", "vtkCoordinate",
           "vtkFileOutputWindow", "vtkFlyingEdges3D", "vtkGPUVolumeRayCastMapper", "vtkImageActor", "vtkImageCast",
           "vtkImageConnectivityFilter", "vtkImageData", "vtkImageGaussianSmooth", "vtkImageReslice",
           "vtkInteractorStyleTrackballCamera", "vtkInteractorStyleUser", "vtkMatrix4x4",
           "vtkOrientationMarkerWidget", "vtkOutputWindow", "vtkPNGWriter", "vtkPiecewiseFunction", "vtkPlanes",
           "vtkPoints", "vtkPolyData", "vtkPolyDataMapper", "vtkPolyDataMapper2D", "vtkPolyDataWriter",
           "vtkPolyLineSource", "vtkRenderWindow", "vtkRenderer", "vtkSTLWriter", "vtkSphereSource", "vtkTextActor",
           "vtkTextRenderer", "vtkTransform", "vtkTransformInterpolator", "vtkTriangleFilter", "vtkTubeFilter",
           "vtkVersion", "vtkVolume", "vtkVolumePicker", "vtkVolumeProperty", "vtkWindowToImageFilter"]
