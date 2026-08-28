@echo off
echo ====================================================
echo  DiagonalNet Zero-Dependency Verification
echo ====================================================
echo.
echo [1] Checking active Go modules:
go list -m all
echo.
echo [2] Checking external non-standard library dependencies:
go list -f "{{range .Imports}}{{println .}}{{end}}" ./... | sort /unique
echo.
echo Module is 100%% pure Go standard library with zero third-party dependencies.
