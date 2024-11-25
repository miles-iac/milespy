Configuration
=============

The following configuration options are available in ``milespy``:

 * ``auto_download``: do not ask for user confirmation for downloading missing repositories.
 * ``repository_folder``: folder where the repositories will be searched and downloaded. By default it is
   the installation forder of ``milespy``.

You can change them by creating a configuration file in any of the following locations
(in order of increasing priority):

 * /etc/milespy/config
 * /etc/milespyrc
 * ~/.config/milespy/config
 * ~/.config/milespy
 * ~/.milespy/config
 * ~/.milespyrc
 * .milespyrc

Or by setting the environment variable ``MILESPY_VAR``, where ``VAR`` is the configuration
option in upper case.
For example ``MILESPY_AUTO_DOWNLOAD=1``.
