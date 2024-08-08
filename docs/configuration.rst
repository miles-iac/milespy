Configuration
=============

The following configuration options are available in ``pymiles``:

 * ``auto_download``: do not ask for user confirmation for downloading missing repositories.
 * ``repository_folder``: folder where the repositories will be searched and downloaded. By default it is
   the installation forder of ``pymiles``.

You can change them by creating a configuration file in any of the following locations
(in order of increasing priority):

 * /etc/pymiles/config
 * /etc/pymilesrc
 * ~/.config/pymiles/config
 * ~/.config/pymiles
 * ~/.pymiles/config
 * ~/.pymilesrc
 * .pymilesrc

Or by setting the environment variable ``PYMILES_VAR``, where ``VAR`` is the configuration
option in upper case.
For example ``PYMILES_AUTO_DOWNLOAD=1``.
