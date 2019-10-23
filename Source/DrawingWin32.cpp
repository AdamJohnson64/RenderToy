#include <exception>

#include <Windows.h>
#include <WinBase.h>

LRESULT WINAPI windowProc(HWND hWnd, UINT Msg, WPARAM wParam, LPARAM lParam)
{
    return DefWindowProc(hWnd, Msg, wParam, lParam);
}

int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ PWSTR lpCmdLine, _In_ int nShowCmd)
{
    try
    {
        {
            WNDCLASS windowClass = {};
            windowClass.lpszClassName = L"DrawingGDI";
            windowClass.lpfnWndProc = windowProc;
            if (!RegisterClass(&windowClass))
                throw std::exception("Failed to register window class.");
        }
        HWND hWnd = {};
        {
            // Calculate the dimensions of the client area at exactly 640x480. 
            RECT windowRect = { 64, 64, 64 + 640, 64 + 480 };
            AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);
            hWnd = CreateWindow(L"DrawingGDI", L"Drawing (Win32)", WS_OVERLAPPEDWINDOW, windowRect.left, windowRect.top, windowRect.right - windowRect.left, windowRect.bottom - windowRect.top, nullptr, nullptr, nullptr, nullptr);
            if (!hWnd)
                throw std::exception("Failed to create window.");
        }
        ShowWindow(hWnd, nShowCmd);
        {
            MSG msg = {};
            while (GetMessage(&msg, hWnd, 0, 0))
                DispatchMessage(&msg);
        }
        return 0;
    }
    catch (...)
    {
        return -1;
    }
}