#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/simon/PathFinderRL/thesis_ws/src/openai_ros/openai_ros"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/simon/PathFinderRL/thesis_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/simon/PathFinderRL/thesis_ws/install/lib/python3/dist-packages:/home/simon/PathFinderRL/thesis_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/simon/PathFinderRL/thesis_ws/build" \
    "/usr/bin/python3" \
    "/home/simon/PathFinderRL/thesis_ws/src/openai_ros/openai_ros/setup.py" \
     \
    build --build-base "/home/simon/PathFinderRL/thesis_ws/build/openai_ros/openai_ros" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/simon/PathFinderRL/thesis_ws/install" --install-scripts="/home/simon/PathFinderRL/thesis_ws/install/bin"
