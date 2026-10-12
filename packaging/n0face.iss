; Inno Setup script for the n0face installer.
; Build with packaging/build.ps1 (it passes /DAppVersion=<version from pyproject>).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6F3B2A51-8E0C-4C3B-9B7A-0F4D2C1E9A77}
AppName=n0face
AppVersion={#AppVersion}
AppVerName=n0face {#AppVersion}
AppPublisher=n0face
AppPublisherURL=https://github.com/kaloyanstefanov66/n0face
DefaultDirName={localappdata}\Programs\n0face
DefaultGroupName=n0face
DisableProgramGroupPage=yes
; Per-user install: no administrator prompt.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=n0face-setup-{#AppVersion}
SetupIconFile=..\assets\n0face.ico
UninstallDisplayIcon={app}\n0face.exe
LicenseFile=..\LICENSE
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\n0face\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\n0face"; Filename: "{app}\n0face.exe"
Name: "{group}\Uninstall n0face"; Filename: "{uninstallexe}"
Name: "{autodesktop}\n0face"; Filename: "{app}\n0face.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\n0face.exe"; Description: "{cm:LaunchProgram,n0face}"; Flags: nowait postinstall skipifsilent
