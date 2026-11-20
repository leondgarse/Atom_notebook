---
tags: [other-languages, flutter]
summary: Flutter SDK install and basic project commands.
---
# ___Flutter___
***

# Basic
## Install
```sh
git clone https://github.com/flutter/flutter.git -b stable
export PATH="$PWD/flutter:$PATH"
```
```sh
flutter clean
flutter pub get
flutter doctor
flutter devices

flutter run -d linux

flutter build apk --release
```
- Use Dart for a flutter application and stitch loop skill for designing.
## Issues
```sh
flutter doctor Doctor summary (to see all details, run flutter doctor -v):
# [✓] Flutter (Channel stable, 3.41.5, on Ubuntu 26.04.1 LTS 7.0.0-1017-oem, locale en_US.UTF-8)
# [✓] Android toolchain - develop for Android devices (Android SDK version 36.1.0)
# [✓] Chrome - develop for the web
# [✗] Linux toolchain - develop for Linux desktop
#     ✗ ninja is required for Linux development.
#       It is likely available from your distribution (e.g.: apt install ninja-build), or can be
#       downloaded from https://github.com/ninja-build/ninja/releases
#     ✗ GTK 3.0 development libraries are required for Linux development.
#       They are likely available from your distribution (e.g.: apt install libgtk-3-dev)
# [✓] Connected device (2 available)
# [✓] Network resources
# ! Doctor found issues in 1 category.

sudo apt update && sudo apt install -y ninja-build libgtk-3-dev
```

```sh
flutter build apk --release
# warning: [options] source value 8 is obsolete and will be removed in a future release
# warning: [options] target value 8 is obsolete and will be removed in a future release
# warning: [options] To suppress warnings about obsolete options, use -Xlint:-options.
# 3 warnings
# 
# FAILURE: Build failed with an exception.
# 
# * What went wrong:
# Execution failed for task ':app:validateSigningRelease'.
# > Keystore file '~/upload-keystore.jks' not found for signing config 'release'.
# 
# * Try:
# > Run with --stacktrace option to get the stack trace.
# > Run with --info or --debug option to get more log output.
# > Run with --scan to get full insights.
# > Get more help at https://help.gradle.org.
# 
# BUILD FAILED in 2m 22s

# Create key
PASSWORD=123456
keytool -genkey -v -keystore /home/leondgarse/upload-keystore.jks -keyalg RSA -keysize 2048 -validity 10000 \
-alias upload -storepass $PASSWORD -keypass $PASSWORD -dname "CN=ServerChat, O=MobileCoder, C=US"
```
<!-- related -->
# Related
- [Go Basic](../2019/12-24_go_basic.md)
- [cgo](../2020/02-11_cgo.md)
***
<!-- /related -->
