#include <QtGui\qimage.h>
#include <QtGui\qpainter.h>
#include <QtWidgets\qapplication.h>
#include <QtWidgets\qwidget.h>
#include <QtWidgets\qmainwindow.h>
#include <Windows.h>

class QRenderToyWindow : public QMainWindow
{
protected:
	void paintEvent(QPaintEvent *event) override
	{
		QPainter painter(this);
		painter.drawLine(0, 0, 100, 100);
		QImage image(256, 256, QImage::Format::Format_ARGB32);
		painter.drawImage(0, 0, image);
	}
};

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	QApplication app(__argc, __argv);
	QRenderToyWindow window;
	window.setFixedWidth(1024);
	window.setFixedHeight(768);
	window.show();
	return app.exec();
}