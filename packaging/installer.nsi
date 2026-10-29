; SoEasy DICOM Viewer - Windows installer (NSIS 3, Modern UI 2).
; Built by `python packaging/build.py --installer`, which runs:
;   makensis /DVERSION=x.y.z /DSRC=<PyInstaller folder> /DOUT=<Setup.exe> /DROOT=<repository> installer.nsi

Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!include "MUI2.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"

!define APP "SoEasy DICOM Viewer"
!define EXE "${APP}.exe"
!define ID "SoEasyDICOMViewer"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${ID}"
!define URL "https://github.com/TrithoneSE/SoEasy-DICOM-Viewer"

Name "${APP} ${VERSION}"
OutFile "${OUT}"
InstallDir "$PROGRAMFILES64\${APP}"
InstallDirRegKey HKLM "${UNINST_KEY}" "InstallLocation"
RequestExecutionLevel admin
BrandingText "${APP} ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${APP}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "${APP} Setup"
VIAddVersionKey "CompanyName" "SoEasy DICOM Viewer contributors"
VIAddVersionKey "LegalCopyright" "GPL-3.0-or-later"

!define MUI_ICON "${ROOT}\assets\icon.ico"
!define MUI_UNICON "${ROOT}\assets\icon.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${ROOT}\packaging\build\wizard.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${ROOT}\packaging\build\wizard.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "${ROOT}\packaging\build\header.bmp"
!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_SMALLDESC

!define MUI_WELCOMEPAGE_TITLE "Welcome to ${APP} ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "This will install ${APP}: 2D, MPR and colour 3D viewing of CT and MRI studies, DSA for angiography runs and DTI tractography.$\r$\n$\r$\n${APP} is a visualisation tool. It is not a certified medical device and must not be used for diagnosis.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN "$INSTDIR\${EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Start ${APP}"
!define MUI_FINISHPAGE_LINK "Documentation and source code"
!define MUI_FINISHPAGE_LINK_LOCATION "${URL}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${ROOT}\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "${APP} needs 64-bit Windows 10 or 11."
    Abort
  ${EndIf}
  SetRegView 64
FunctionEnd

Function un.onInit
  SetRegView 64
FunctionEnd

Section "!${APP}" SecApp
  SectionIn RO
  SetOutPath "$INSTDIR"
  RMDir /r "$INSTDIR\_internal"            ; an update must not keep libraries of the previous version
  File /r "${SRC}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  CreateDirectory "$SMPROGRAMS\${APP}"
  CreateShortcut "$SMPROGRAMS\${APP}\${APP}.lnk" "$INSTDIR\${EXE}"
  CreateShortcut "$SMPROGRAMS\${APP}\${APP} (safe graphics).lnk" "$INSTDIR\${EXE}" "--software-gl" \
    "$INSTDIR\${EXE}" 0 SW_SHOWNORMAL "" "Start with software 3D rendering (graphics driver problems)"

  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "${APP}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "SoEasy DICOM Viewer contributors"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\${EXE}"
  WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UNINST_KEY}" "URLInfoAbout" "${URL}"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${UNINST_KEY}" "EstimatedSize" $0
SectionEnd

Section "Desktop shortcut" SecDesktop
  CreateShortcut "$DESKTOP\${APP}.lnk" "$INSTDIR\${EXE}"
SectionEnd

Section "Explorer: Open folder in ${APP}" SecFolder
  WriteRegStr HKLM "Software\Classes\Directory\shell\${ID}" "" "Open in ${APP}"
  WriteRegStr HKLM "Software\Classes\Directory\shell\${ID}" "Icon" "$INSTDIR\${EXE}"
  WriteRegStr HKLM "Software\Classes\Directory\shell\${ID}\command" "" '"$INSTDIR\${EXE}" "%1"'
SectionEnd

Section "Explorer: .dcm files in Open with" SecDcm
  WriteRegStr HKLM "Software\Classes\${ID}.dcm" "" "DICOM image"
  WriteRegStr HKLM "Software\Classes\${ID}.dcm\DefaultIcon" "" "$INSTDIR\${EXE},0"
  WriteRegStr HKLM "Software\Classes\${ID}.dcm\shell\open\command" "" '"$INSTDIR\${EXE}" "%1"'
  WriteRegStr HKLM "Software\Classes\.dcm\OpenWithProgids" "${ID}.dcm" ""
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecApp} "The viewer itself (required)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "A shortcut on the desktop."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecFolder} "Right-click a folder (CD, USB stick, download) and open the DICOM studies in it."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDcm} "Adds ${APP} to Open with for .dcm files. Your current default program stays."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  Delete "$DESKTOP\${APP}.lnk"
  RMDir /r "$SMPROGRAMS\${APP}"
  RMDir /r "$INSTDIR\_internal"
  Delete "$INSTDIR\${EXE}"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKLM "Software\Classes\Directory\shell\${ID}"
  DeleteRegValue HKLM "Software\Classes\.dcm\OpenWithProgids" "${ID}.dcm"
  DeleteRegKey HKLM "Software\Classes\${ID}.dcm"
  DeleteRegKey HKLM "${UNINST_KEY}"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
  ; colour presets and logs in %APPDATA%\DicomColorViewer are the user's and are kept
SectionEnd
