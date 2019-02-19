//------------------------------------------------------------------------------
// Copyright (c) 2019 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------
/*
This file is from GitHub: https://github.com/tniessen/iperf-windows/blob/master/win32-compat/sys/utsname.h
*/

#ifndef UTSNAME_H_
#define UTSNAME_H_

#define WIN32_LEAN_AND_MEAN
#include <WinSock2.h>
#include <Windows.h>
#include <stdio.h>

#define UTSNAME_MAXLENGTH 256

struct utsname {
	char sysname [UTSNAME_MAXLENGTH]; // name of this implementation of the operating system
	char nodename[UTSNAME_MAXLENGTH]; // name of this node within an implementation - dependent communications network
	char release [UTSNAME_MAXLENGTH]; //  current release level of this implementation
	char version [UTSNAME_MAXLENGTH]; //  current version level of this release
	char machine [UTSNAME_MAXLENGTH]; //  name of the hardware type on which the system is running
};

int uname(struct utsname *name);

int uname(struct utsname *name) {
	OSVERSIONINFO versionInfo;
	SYSTEM_INFO sysInfo;

	// Get Windows version info
	ZeroMemory(&versionInfo, sizeof(OSVERSIONINFO));
	versionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx(&versionInfo);

	// Get hardware info
	ZeroMemory(&sysInfo, sizeof(SYSTEM_INFO));
	GetSystemInfo(&sysInfo);

	// Set implementation name
	strcpy(name->sysname, "Windows");
	itoa(versionInfo.dwBuildNumber, name->release, 10);
	sprintf(name->version, "%i.%i", versionInfo.dwMajorVersion, versionInfo.dwMinorVersion);

	// Set hostname
	if (gethostname(name->nodename, UTSNAME_MAXLENGTH) != 0) {
		return WSAGetLastError();
	}

	// Set processor architecture
	switch (sysInfo.wProcessorArchitecture) {
	case PROCESSOR_ARCHITECTURE_AMD64:
		strcpy(name->machine, "x86_64");
		break;
	case PROCESSOR_ARCHITECTURE_IA64:
		strcpy(name->machine, "ia64");
		break;
	case PROCESSOR_ARCHITECTURE_INTEL:
		strcpy(name->machine, "x86");
		break;
	case PROCESSOR_ARCHITECTURE_UNKNOWN:
	default:
		strcpy(name->machine, "unknown");
	}

	return 0;
}

#endif // UTSNAME_H_
