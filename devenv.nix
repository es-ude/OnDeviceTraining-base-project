{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.pico-sdk
    pkgs.picotool
    pkgs.cmake
    # pkgs.gcc
    pkgs.clang-tools
    pkgs.gcc-arm-embedded
    pkgs.picotool
    pkgs.minicom
    pkgs.ninja
];

  languages = {
    python = {
      enable = true;
      package = pkgs.python313;
    };
    c = {
      enable = true;
    };
  };

  # https://devenv.sh/languages/
  # languages.rust.enable = true;

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;


  scripts.open-serial-console.exec = ''
    minicom --device /dev/ttyACM0 
  '';

  # See full reference at https://devenv.sh/reference/options/
}
