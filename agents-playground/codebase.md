# .eslintrc.json

```json
{
  "extends": "next/core-web-vitals"
}

```

# .github\banner_dark.png

This is a binary file of the type: Image

# .github\banner_light.png

This is a binary file of the type: Image

# .github\ISSUE_TEMPLATE\bug_report.yaml

```yaml
name: "\U0001F41E Bug report"
description: Report an issue with LiveKit Agent Playground
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
        Please report security issues by email to security@livekit.io
  - type: textarea
    id: bug-description
    attributes:
      label: Describe the bug
      description: Describe what you are expecting vs. what happens instead.
      placeholder: |
        ### What I'm expecting
        ### What happens instead
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction
      description: A detailed step-by-step guide on how to reproduce the issue or (preferably) a link to a repository that reproduces the issue. Reproductions must be [short, self-contained and correct](http://sscce.org/) and must not contain files or code that aren't relevant to the issue. It's best if you use the sample app in `example/index.ts` as a starting point for your reproduction. We will prioritize issues that include a working minimal reproduction repository.
      placeholder: Reproduction
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Logs
      description: "Please include browser console and server logs around the time this bug occurred. Enable debug logging by calling `setLogLevel('debug')` from the livekit-client package as early as possible. Please try not to insert an image but copy paste the log text."
      render: shell
  - type: textarea
    id: system-info
    attributes:
      label: System Info
      description: Please mention the OS (incl. Version) and Browser (including exact version) on which you are seeing this issue. For ease of use you can run `npx envinfo --system --binaries --browsers --npmPackages "{livekit-client, @livekit/*}"` from within your livekit project, to give us all the needed info about your current environment
      render: shell
      placeholder: System, Binaries, Browsers
    validations:
      required: true
  - type: dropdown
    id: severity
    attributes:
      label: Severity
      options:
        - annoyance
        - serious, but I can work around it
        - blocking an upgrade
        - blocking all usage of LiveKit
    validations:
      required: true
  - type: textarea
    id: additional-context
    attributes:
      label: Additional Information

```

# .github\ISSUE_TEMPLATE\config.yml

```yml
blank_issues_enabled: false
contact_links:
  - name: Slack Community Chat
    url: https://livekit.io/join-slack
    about: Ask questions and discuss with other LiveKit users in real time.
```

# .github\ISSUE_TEMPLATE\feature_request.yaml

```yaml
name: "Feature Request"
description: Suggest an idea for this project
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to request this feature!
  - type: textarea
    id: problem
    attributes:
      label: Describe the problem
      description: Please provide a clear and concise description the problem this feature would solve. The more information you can provide here, the better.
      placeholder: I would like to be able to ... in order to solve ...
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Describe the proposed solution
      description: Please provide a clear and concise description of what you would like to happen.
      placeholder: I would like to see ...
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
      description: "Please provide a clear and concise description of any alternative solutions or features you've considered."
  - type: dropdown
    id: importance
    attributes:
      label: Importance
      description: How important is this feature to you?
      options:
        - nice to have
        - would make my life easier
        - I cannot use LiveKit without it
    validations:
      required: true
  - type: textarea
    id: additional-context
    attributes:
      label: Additional Information
      description: Add any other context or screenshots about the feature request here.
```

# .gitignore

```


```

# LICENSE

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```

# next-env.d.ts

```ts
/// <reference types="next" />
/// <reference types="next/image-types/global" />

// NOTE: This file should not be edited
// see https://nextjs.org/docs/pages/building-your-application/configuring/typescript for more information.

```

# next.config.js

```js
const createNextPluginPreval = require("next-plugin-preval/config");
const withNextPluginPreval = createNextPluginPreval();

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
};

module.exports = withNextPluginPreval(nextConfig);

```

# NOTICE

```
Copyright 2024 LiveKit, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

# package.json

```json
{
  "name": "agents-playground",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@livekit/components-react": "^2.6.0",
    "@livekit/components-styles": "^1.1.1",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "cookies-next": "^4.1.1",
    "framer-motion": "^10.16.16",
    "js-yaml": "^4.1.0",
    "livekit-client": "^2.5.1",
    "livekit-server-sdk": "^2.6.1",
    "lodash": "^4.17.21",
    "next": "^14.0.4",
    "next-plugin-preval": "^1.2.6",
    "qrcode.react": "^4.0.0",
    "react": "^18",
    "react-dom": "^18",
    "react-virtuoso": "^4.12.5"
  },
  "devDependencies": {
    "@types/js-yaml": "^4.0.9",
    "@types/lodash": "^4.17.0",
    "@types/node": "^20.10.4",
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.4.16",
    "eslint": "^8",
    "eslint-config-next": "14.2.15",
    "postcss": "^8.4.31",
    "tailwindcss": "^3.3.5",
    "typescript": "^5.3.3"
  }
}

```

# postcss.config.js

```js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}

```

# public\favicon.ico

This is a binary file of the type: Binary

# public\logo.svg

This is a file of the type: SVG Image

# public\next.svg

This is a file of the type: SVG Image

# public\vercel.svg

This is a file of the type: SVG Image

# README.md

```md
<!--BEGIN_BANNER_IMAGE-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/.github/banner_light.png">
  <img style="width:100%;" alt="The LiveKit icon, the name of the repository and some sample code in the background." src="https://raw.githubusercontent.com/livekit/agent-playground/main/.github/banner_light.png">
</picture>

<!--END_BANNER_IMAGE-->

# LiveKit Agents Playground

<!--BEGIN_DESCRIPTION-->
The Agent Playground is designed for quickly prototyping with server side agents built with [LiveKit Agents Framework](https://github.com/livekit/agents). Easily tap into LiveKit WebRTC sessions and process or generate audio, video, and data streams.
  The playground includes components to fully interact with any LiveKit agent, through video, audio and chat.
<!--END_DESCRIPTION-->

## Docs and references

Docs for how to get started with LiveKit agents at [https://docs.livekit.io/agents](https://docs.livekit.io/agents)

The repo containing the (server side) agent implementations (including example agents): [https://github.com/livekit/agents](https://github.com/livekit/agents)

## Try out a live version

You can try out the agents playground at [https://livekit-agent-playground.vercel.app](https://livekit-agent-playground.vercel.app).
This will connect you to our example agent, KITT, which is based off of the [minimal-assistant](https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/minimal_assistant.py).

## Setting up the playground locally

1. Install dependencies

\`\`\`bash
  npm install
\`\`\`

2. Copy and rename the `.env.example` file to `.env.local` and fill in the necessary environment variables.

\`\`\`
LIVEKIT_API_KEY=<your API KEY>
LIVEKIT_API_SECRET=<Your API Secret>
NEXT_PUBLIC_LIVEKIT_URL=wss://<Your Cloud URL>
\`\`\`

3. Run the development server:

\`\`\`bash
  npm run dev
\`\`\`

4. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
5. If you haven't done so yet, start your agent (with the same project variables as in step 2.)
6. Connect to a room and see your agent connecting to the playground

## Features

- Render video, audio and chat from your agent
- Send video, audio, or text to your agent
- Configurable settings panel to work with your agent

## Notes

- This playground is currently work in progress. There are known layout/responsive bugs and some features are under tested.
- The playground was tested against the kitt example in `https://github.com/livekit/agents`.
- Feel free to ask questions, request features in our [community slack](https://livekit.io/join-slack).

## Known issues

- Layout can break on smaller screens.
- Mobile device sizes not supported currently

<!--BEGIN_REPO_NAV-->
<br/><table>
<thead><tr><th colspan="2">LiveKit Ecosystem</th></tr></thead>
<tbody>
<tr><td>LiveKit SDKs</td><td><a href="https://github.com/livekit/client-sdk-js">Browser</a> · <a href="https://github.com/livekit/client-sdk-swift">iOS/macOS/visionOS</a> · <a href="https://github.com/livekit/client-sdk-android">Android</a> · <a href="https://github.com/livekit/client-sdk-flutter">Flutter</a> · <a href="https://github.com/livekit/client-sdk-react-native">React Native</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/client-sdk-unity">Unity</a> · <a href="https://github.com/livekit/client-sdk-unity-web">Unity (WebGL)</a></td></tr><tr></tr>
<tr><td>Server APIs</td><td><a href="https://github.com/livekit/node-sdks">Node.js</a> · <a href="https://github.com/livekit/server-sdk-go">Golang</a> · <a href="https://github.com/livekit/server-sdk-ruby">Ruby</a> · <a href="https://github.com/livekit/server-sdk-kotlin">Java/Kotlin</a> · <a href="https://github.com/livekit/python-sdks">Python</a> · <a href="https://github.com/livekit/rust-sdks">Rust</a> · <a href="https://github.com/agence104/livekit-server-sdk-php">PHP (community)</a> · <a href="https://github.com/pabloFuente/livekit-server-sdk-dotnet">.NET (community)</a></td></tr><tr></tr>
<tr><td>UI Components</td><td><a href="https://github.com/livekit/components-js">React</a> · <a href="https://github.com/livekit/components-android">Android Compose</a> · <a href="https://github.com/livekit/components-swift">SwiftUI</a></td></tr><tr></tr>
<tr><td>Agents Frameworks</td><td><a href="https://github.com/livekit/agents">Python</a> · <a href="https://github.com/livekit/agents-js">Node.js</a> · <b>Playground</b></td></tr><tr></tr>
<tr><td>Services</td><td><a href="https://github.com/livekit/livekit">LiveKit server</a> · <a href="https://github.com/livekit/egress">Egress</a> · <a href="https://github.com/livekit/ingress">Ingress</a> · <a href="https://github.com/livekit/sip">SIP</a></td></tr><tr></tr>
<tr><td>Resources</td><td><a href="https://docs.livekit.io">Docs</a> · <a href="https://github.com/livekit-examples">Example apps</a> · <a href="https://livekit.io/cloud">Cloud</a> · <a href="https://docs.livekit.io/home/self-hosting/deployment">Self-hosting</a> · <a href="https://github.com/livekit/livekit-cli">CLI</a></td></tr>
</tbody>
</table>
<!--END_REPO_NAV-->

```

# renovate.json

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:recommended"],
  "packageRules": [
    {
      "schedule": "before 6am on the first day of the month",
      "matchDepTypes": ["devDependencies"],
      "matchUpdateTypes": ["patch", "minor"],
      "groupName": "devDependencies (non-major)"
    },
    {
      "matchSourceUrlPrefixes": ["https://github.com/livekit/"],
      "matchUpdateTypes": ["patch", "minor"],
      "groupName": "LiveKit dependencies (non-major)",
      "automerge": true
    }
  ]
}

```

# src\cloud\CloudConnect.tsx

```tsx
export const CloudConnect = ({ accentColor }: { accentColor: string }) => {
  return null;
};

export const CLOUD_ENABLED = false;

```

# src\cloud\README.md

```md
Files in this `cloud/` directory can be ignored. They are mocks which we override in our private, hosted version of the agents-playground that supports LiveKit Cloud authentication.
```

# src\cloud\useCloud.tsx

```tsx
export function CloudProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export function useCloud() {
  const generateToken: () => Promise<string> = async () => {
    throw new Error("Not implemented");
  };
  const wsUrl = "";

  return { generateToken, wsUrl };
}
```

# src\components\button\Button.tsx

```tsx
import React, { ButtonHTMLAttributes, ReactNode, useEffect, useRef } from "react";
import { addGlowTrail, addDistortionPulse } from "@/lib/animations";

type ButtonProps = {
  accentColor: string;
  children: ReactNode;
  className?: string;
  disabled?: boolean;
} & ButtonHTMLAttributes<HTMLButtonElement>;

export const Button: React.FC<ButtonProps> = ({
  accentColor,
  children,
  className = "",
  disabled = false,
  ...allProps
}) => {
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (buttonRef.current && !disabled) {
      // Add interactive effects
      addGlowTrail(buttonRef.current);
      addDistortionPulse(buttonRef.current);
    }
  }, [disabled]);

  return (
    <button
      ref={buttonRef}
      className={`relative flex flex-row ${
        disabled ? "pointer-events-none opacity-50" : ""
      } text-gray-200 text-sm justify-center border border-${accentColor}-500/40 px-6 py-2 rounded-md transition-all duration-300 
      hover:shadow-lg hover:shadow-${accentColor}-500/30 hover:scale-105 
      active:scale-[0.98] ${className}`}
      disabled={disabled}
      {...allProps}
    >
      {/* Neon gradient background */}
      <span 
        className={`absolute inset-0 bg-gradient-to-r from-${accentColor}-500/10 via-transparent to-${accentColor}-500/10 rounded-md ${
          disabled ? "" : "animate-pulse"
        }`}
      ></span>
      
      {/* Button content */}
      <span className="relative z-10 flex items-center gap-2">
        {children}
      </span>
    </button>
  );
};
```

# src\components\button\LoadingSVG.tsx

```tsx
export const LoadingSVG = ({
  diameter = 20,
  strokeWidth = 4,
}: {
  diameter?: number;
  strokeWidth?: number;
}) => (
  <svg
    className="animate-spin"
    fill="none"
    viewBox="0 0 24 24"
    style={{
      width: `${diameter}px`,
      height: `${diameter}px`,
    }}
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth={strokeWidth}
    ></circle>
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    ></path>
  </svg>
);

```

# src\components\chat\ChatMessage.tsx

```tsx
import React, { useEffect, useRef, memo } from "react";
import { addScanLineEffect } from "@/lib/animations";

type ChatMessageProps = {
  message: string;
  accentColor: string;
  name: string;
  isSelf: boolean;
  hideName?: boolean;
};

export const ChatMessage = memo(({
  name,
  message,
  accentColor,
  isSelf,
  hideName,
}: ChatMessageProps) => {
  const messageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messageRef.current) {
      // Add scan line effect to message
      const cleanupScanLine = addScanLineEffect(messageRef.current);
      return () => cleanupScanLine();
    }
  }, []);

  return (
    <div 
      className={`flex flex-col gap-1 ${hideName ? "pt-0" : "pt-4"}`}
    >
      {!hideName && (
        <div
          className={`text-${
            isSelf ? "gray-500" : accentColor + "-400"
          } uppercase text-xs tracking-wider`}
        >
          {name}
        </div>
      )}
      
      <div
        ref={messageRef}
        className={`
          relative glass-panel p-3 
          ${isSelf ? "bg-gray-900/30" : `bg-${accentColor}-900/20`}
          border ${isSelf ? "border-gray-700/30" : `border-${accentColor}-500/30`}
          shadow-sm ${isSelf ? "" : `shadow-${accentColor}-500/20`}
          hover:shadow-md hover:shadow-${accentColor}-500/30 
          transition-all duration-300
          max-w-[90%] ${isSelf ? "ml-auto" : "mr-auto"}
          overflow-hidden
        `}
      >
        <div
          className={`pr-4 text-${
            isSelf ? "gray-300" : accentColor + "-300"
          } text-sm whitespace-pre-line`}
        >
          {message}
        </div>
        
        {/* Holographic overlay */}
        <div 
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `linear-gradient(90deg, 
              transparent 0%, 
              rgba(${isSelf ? "255, 255, 255" : "0, 255, 255"}, 0.05) 25%, 
              transparent 50%
            )`,
            opacity: "0.3"
          }}
        ></div>
      </div>
    </div>
  );
});
```

# src\components\chat\ChatMessageInput.tsx

```tsx
import { useWindowResize } from "@/hooks/useWindowResize";
import { useCallback, useEffect, useRef, useState } from "react";

type ChatMessageInputProps = {
  placeholder: string;
  accentColor: string;
  height: number;
  onSend?: (message: string) => void;
};

export const ChatMessageInput: React.FC<ChatMessageInputProps> = ({
  placeholder,
  accentColor,
  height,
  onSend,
}) => {
  const [message, setMessage] = useState("");
  const [inputTextWidth, setInputTextWidth] = useState(0);
  const [inputWidth, setInputWidth] = useState(0);
  const hiddenInputRef = useRef<HTMLSpanElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const windowSize = useWindowResize();
  const [isTyping, setIsTyping] = useState(false);
  const [inputHasFocus, setInputHasFocus] = useState(false);

  const handleSend = useCallback(() => {
    if (!onSend || message === "") {
      return;
    }

    onSend(message);
    setMessage("");
  }, [onSend, message]);

  useEffect(() => {
    setIsTyping(true);
    const timeout = setTimeout(() => {
      setIsTyping(false);
    }, 500);

    return () => clearTimeout(timeout);
  }, [message]);

  useEffect(() => {
    if (hiddenInputRef.current) {
      setInputTextWidth(hiddenInputRef.current.clientWidth);
    }
  }, [hiddenInputRef, message]);

  useEffect(() => {
    if (inputRef.current) {
      setInputWidth(inputRef.current.clientWidth);
    }
  }, [hiddenInputRef, message, windowSize.width]);

  return (
    <div
      className="flex flex-col gap-2 border-t border-t-gray-800 relative glass-panel mt-2"
      style={{ height }}
    >
      {/* Scan line effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none scan-line"></div>
      
      <div className="flex flex-row pt-3 gap-2 items-center relative">
        {/* Digital cursor */}
        <div
          className={`
            w-2 h-4 
            bg-${inputHasFocus ? accentColor : "gray"}-${inputHasFocus ? "400" : "800"} 
            ${inputHasFocus ? "border-glow" : ""} 
            absolute left-2 
            ${!isTyping && inputHasFocus ? "cursor-animation" : ""}
            transition-all duration-200
          `}
          style={{
            transform: `translateX(${
              message.length > 0
                ? Math.min(inputTextWidth, inputWidth - 20) - 4
                : 0
            }px)`,
          }}
        ></div>
        
        {/* Input field */}
        <input
          ref={inputRef}
          className={`
            w-full text-sm caret-transparent bg-transparent 
            text-${accentColor}-100 p-2 pr-6 rounded-sm 
            focus:outline-none focus:ring-1 focus:ring-${accentColor}-700
            transition-all duration-300
            backdrop-blur-sm
            ${inputHasFocus ? "bg-gray-900/20" : "bg-gray-900/10"}
          `}
          style={{
            paddingLeft: message.length > 0 ? "12px" : "24px",
            caretShape: "block",
          }}
          placeholder={placeholder}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
          }}
          onFocus={() => {
            setInputHasFocus(true);
          }}
          onBlur={() => {
            setInputHasFocus(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              handleSend();
            }
          }}
        ></input>
        
        {/* Hidden element to measure text width */}
        <span
          ref={hiddenInputRef}
          className="absolute top-0 left-0 text-sm pl-3 text-amber-500 pointer-events-none opacity-0"
        >
          {message.replaceAll(" ", "\u00a0")}
        </span>
        
        {/* Send button */}
        <button
          disabled={message.length === 0 || !onSend}
          onClick={handleSend}
          className={`
            text-xs uppercase tracking-wider
            text-${accentColor}-400 hover:text-${accentColor}-300
            hover:bg-${accentColor}-900/20 p-2 rounded-md
            opacity-${message.length > 0 ? "100" : "25"}
            pointer-events-${message.length > 0 ? "auto" : "none"}
            transition-all duration-300
            border border-transparent hover:border-${accentColor}-500/30
          `}
        >
          Send
        </button>
      </div>
    </div>
  );
};
```

# src\components\chat\ChatTile.tsx

```tsx
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatMessageInput } from "@/components/chat/ChatMessageInput";
import { ChatMessage as ComponentsChatMessage } from "@livekit/components-react";
import { useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Virtuoso } from 'react-virtuoso';

const inputHeight = 48;

export type ChatMessageType = {
  name: string;
  message: string;
  isSelf: boolean;
  timestamp: number;
};

type ChatTileProps = {
  messages: ChatMessageType[];
  accentColor: string;
  onSend?: (message: string) => Promise<ComponentsChatMessage>;
};

export const ChatTile = ({ messages, accentColor, onSend }: ChatTileProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Memoize message rendering function
  const renderMessage = useMemo(() => (message: ChatMessageType, index: number) => {
    const hideName = index >= 1 && messages[index - 1].name === message.name;
    
    return (
      <motion.div
        key={`${message.name}-${message.timestamp}-${index}`}
        initial={{ opacity: 0, y: 10, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, scale: 0.98 }}
        transition={{ 
          duration: 0.2,
          type: "spring",
          stiffness: 400,
          damping: 25
        }}
      >
        <ChatMessage
          hideName={hideName}
          name={message.name}
          message={message.message}
          isSelf={message.isSelf}
          accentColor={accentColor}
        />
      </motion.div>
    );
  }, [messages, accentColor]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [containerRef, messages]);

  return (
    <div className="flex flex-col gap-2 w-full h-full relative">
      {/* Holographic grid overlay */}
      <div className="absolute inset-0 pointer-events-none holo-grid opacity-10"></div>
      
      <div
        ref={containerRef}
        className="overflow-y-auto scrollbar-thin scrollbar-track-gray-900/20 scrollbar-thumb-cyan-500/20"
        style={{
          height: `calc(100% - ${inputHeight}px)`,
        }}
      >
        <Virtuoso
          style={{ height: '100%' }}
          data={messages}
          itemContent={(index, message) => renderMessage(message, index)}
          followOutput="smooth"
          alignToBottom
        />
      </div>
      
      <ChatMessageInput
        height={inputHeight}
        placeholder="Type a message"
        accentColor={accentColor}
        onSend={onSend ? (message) => onSend(message) : undefined}
      />
    </div>
  );
};
```

# src\components\colorPicker\ColorPicker.tsx

```tsx
import { useState, useEffect, useRef } from "react";

type ColorPickerProps = {
  colors: string[];
  selectedColor: string;
  onSelect: (color: string) => void;
};

export const ColorPicker = ({
  colors,
  selectedColor,
  onSelect,
}: ColorPickerProps) => {
  const [isHovering, setIsHovering] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const onMouseEnter = () => {
    setIsHovering(true);
  };
  
  const onMouseLeave = () => {
    setIsHovering(false);
  };

  // Add a pulse effect to the selected color
  useEffect(() => {
    if (containerRef.current) {
      const selectedEl = containerRef.current.querySelector(
        `[data-color="${selectedColor}"]`
      ) as HTMLDivElement;
      
      if (selectedEl) {
        const pulseInterval = setInterval(() => {
          selectedEl.style.boxShadow = "0 0 12px var(--neon-cyan)";
          
          setTimeout(() => {
            selectedEl.style.boxShadow = "0 0 6px var(--neon-cyan)";
          }, 500);
        }, 2000);
        
        return () => clearInterval(pulseInterval);
      }
    }
  }, [selectedColor]);

  return (
    <div
      ref={containerRef}
      className="flex flex-row gap-2 py-3 flex-wrap glass-panel p-4"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {colors.map((color) => {
        const isSelected = color === selectedColor;
        const saturation = !isHovering && !isSelected ? "saturate-[0.5] opacity-40" : "";
        const borderColor = isSelected
          ? `border border-${color}-400`
          : "border-gray-800/40";
          
        return (
          <div
            key={color}
            data-color={color}
            className={`
              ${saturation} rounded-md p-1 border-2 ${borderColor}
              cursor-pointer hover:opacity-100 transition-all duration-300
              hover:scale-110 hover:shadow-lg hover:shadow-${color}-500/30
              ${isSelected ? "scale-110 shadow-lg shadow-${color}-500/30" : ""}
            `}
            onClick={() => {
              onSelect(color);
            }}
          >
            <div 
              className={`w-6 h-6 bg-${color}-500 rounded-sm`}
              style={{
                boxShadow: isSelected ? "0 0 8px rgba(0, 255, 255, 0.5)" : "none",
              }}
            ></div>
          </div>
        );
      })}
    </div>
  );
};
```

# src\components\config\AudioInputTile.tsx

```tsx
import {
  BarVisualizer,
  TrackReferenceOrPlaceholder,
} from "@livekit/components-react";

export const AudioInputTile = ({
  trackRef,
}: {
  trackRef: TrackReferenceOrPlaceholder;
}) => {
  return (
    <div
      className={`flex flex-row gap-2 h-[100px] items-center w-full justify-center border rounded-sm border-gray-800 bg-gray-900`}
    >
      <BarVisualizer
        trackRef={trackRef}
        className="h-full w-full"
        barCount={20}
        options={{ minHeight: 0 }}
      />
    </div>
  );
};

```

# src\components\config\ConfigurationPanelItem.tsx

```tsx
import { ReactNode, useRef, useEffect } from "react";
import { PlaygroundDeviceSelector } from "@/components/playground/PlaygroundDeviceSelector";
import { TrackToggle } from "@livekit/components-react";
import { Track } from "livekit-client";
import { createTextFlicker } from "@/lib/animations";

type ConfigurationPanelItemProps = {
  title: string;
  children?: ReactNode;
  deviceSelectorKind?: MediaDeviceKind;
};

export const ConfigurationPanelItem: React.FC<ConfigurationPanelItemProps> = ({
  children,
  title,
  deviceSelectorKind,
}) => {
  const titleRef = useRef<HTMLHeadingElement>(null);
  
  useEffect(() => {
    if (titleRef.current) {
      // Add flickering effect to title for cyberpunk feel
      createTextFlicker(titleRef.current);
    }
  }, []);
  
  return (
    <div className="w-full text-gray-300 py-4 border-b border-b-gray-800/50 relative hud-panel mb-4">
      {/* Top edge cyberpunk accent */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-cyan-500/0 via-cyan-500/50 to-cyan-500/0"></div>
      
      <div className="flex flex-row justify-between items-center px-4 text-xs uppercase tracking-wider">
        <h3 
          ref={titleRef}
          className="text-cyan-400 font-mono relative text-glow tracking-widest py-1"
        >
          {title}
          
          {/* Line decoration under title */}
          <div className="absolute bottom-0 left-0 w-1/2 h-[1px] bg-cyan-500/30"></div>
        </h3>
        
        {deviceSelectorKind && (
          <span className="flex flex-row gap-2">
            <TrackToggle
              className="px-2 py-1 bg-gray-900/50 text-gray-300 border border-gray-800 rounded-sm hover:bg-gray-800/70 hover:border-cyan-500/30 transition-all"
              source={
                deviceSelectorKind === "audioinput"
                  ? Track.Source.Microphone
                  : Track.Source.Camera
              }
            />
            <PlaygroundDeviceSelector kind={deviceSelectorKind} />
          </span>
        )}
      </div>
      <div className="px-4 py-3 text-sm text-gray-400 leading-normal">
        {children}
      </div>
    </div>
  );
};
```

# src\components\config\NameValueRow.tsx

```tsx
import { ReactNode } from "react";

type NameValueRowProps = {
  name: string;
  value?: ReactNode;
  valueColor?: string;
};

export const NameValueRow: React.FC<NameValueRowProps> = ({
  name,
  value,
  valueColor = "cyan-400",
}) => {
  return (
    <div className="flex flex-row w-full items-baseline text-sm py-1 border-b border-gray-800/30">
      <div className="grow shrink-0 text-gray-500 font-mono tracking-wider text-xs uppercase">
        {name}
      </div>
      <div className={`text-sm shrink text-${valueColor} text-right font-mono tracking-wide digital-flicker`}>
        {value}
      </div>
    </div>
  );
};
```

# src\components\cyberpunk\NeuralInterfaceAnimation.tsx

```tsx
import React from 'react';

export const NeuralInterfaceAnimation: React.FC = () => {
  return (
    <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" className="w-full h-full" style={{ backgroundColor: 'black' }}>
        {/* Definitions for reusable elements */}
        <defs>
          <linearGradient id="scanline" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(0,255,255,0)" />
            <stop offset="50%" stopColor="rgba(0,255,255,0.1)" />
            <stop offset="100%" stopColor="rgba(0,255,255,0)" />
          </linearGradient>
          
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          
          <radialGradient id="pulseGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.7)" />
            <stop offset="40%" stopColor="rgba(255,255,255,0.4)" />
            <stop offset="100%" stopColor="rgba(0,255,255,0)" />
            <animate attributeName="r" values="40%;60%;40%" dur="3s" repeatCount="indefinite" />
          </radialGradient>
          
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(0,255,255,0.1)" />
            <stop offset="50%" stopColor="rgba(0,255,255,0.8)" />
            <stop offset="100%" stopColor="rgba(0,255,255,0.1)" />
            <animate attributeName="x1" values="0%;100%;0%" dur="4s" repeatCount="indefinite" />
            <animate attributeName="x2" values="100%;200%;100%" dur="4s" repeatCount="indefinite" />
          </linearGradient>
          
          <symbol id="triangleMarker" viewBox="0 0 20 20">
            <path d="M10,2 L18,18 L2,18 Z" fill="white" />
          </symbol>
          
          <symbol id="circleMarker" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" stroke="white" strokeWidth="1" fill="none" />
            <circle cx="10" cy="10" r="2" fill="white" />
          </symbol>
          
          <symbol id="targetMarker" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" stroke="white" strokeWidth="1" fill="none" />
            <circle cx="10" cy="10" r="1" fill="white" />
            <line x1="10" y1="2" x2="10" y2="6" stroke="white" strokeWidth="1" />
            <line x1="10" y1="14" x2="10" y2="18" stroke="white" strokeWidth="1" />
            <line x1="2" y1="10" x2="6" y2="10" stroke="white" strokeWidth="1" />
            <line x1="14" y1="10" x2="18" y2="10" stroke="white" strokeWidth="1" />
          </symbol>
          
          <symbol id="dotMarker" viewBox="0 0 10 10">
            <circle cx="5" cy="5" r="2" fill="white" />
          </symbol>
        </defs>
        
        {/* Background grid lines */}
        <g id="gridLines" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
          {[...Array(9)].map((_, i) => (
            <line key={`v${i}`} x1={(i + 1) * 100} y1="0" x2={(i + 1) * 100} y2="600" />
          ))}
          {[...Array(5)].map((_, i) => (
            <line key={`h${i}`} x1="0" y1={(i + 1) * 100} x2="1000" y2={(i + 1) * 100} strokeOpacity="0.05" />
          ))}
        </g>
        
        {/* Coordinate lines */}
        <g id="diagonalLines" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
          <line x1="0" y1="150" x2="1000" y2="450" />
          <line x1="250" y1="0" x2="600" y2="600" stroke="url(#lineGradient)" />
          <line x1="900" y1="100" x2="400" y2="500" />
        </g>
        
        {/* Scan line effect - moving down */}
        <rect id="scanLine" x="0" y="0" width="1000" height="20" fill="url(#scanline)" opacity="0.1">
          <animate attributeName="y" values="0;600;0" dur="8s" repeatCount="indefinite" />
        </rect>
        
        {/* Japanese text labels */}
        <g fontFamily="monospace" fontSize="14" fill="rgba(255,255,255,0.6)">
          <text x="330" y="200" id="nonte1">ノンテ</text>
          <text x="550" y="330" id="nonte2">ノンテ</text>
          <text x="750" y="250" id="nonte3">ノンテ</text>
          <text x="180" y="460" id="nonte4">ノンテ</text>
          <text x="600" y="550" id="nonte5">ノンテ</text>
          <text x="850" y="500" id="nonte6">ノンテ</text>
          <text x="765" y="730" id="nonte7">ノンテ</text>
          
          <animate xlinkHref="#nonte2" attributeName="opacity" values="0.6;0.1;0.6" dur="3s" begin="1s" repeatCount="indefinite" />
          <animate xlinkHref="#nonte5" attributeName="opacity" values="0.6;0.1;0.6" dur="4s" begin="0s" repeatCount="indefinite" />
        </g>
        
        {/* Numerical indicators */}
        <g fontFamily="monospace" fontSize="12" fill="rgba(255,255,255,0.8)">
          <text x="30" y="105">761</text>
          <text x="960" y="105">761</text>
          <text x="30" y="482">796</text>
          <text x="960" y="482">796</text>
          <text x="30" y="600">716</text>
          <text x="960" y="600">716</text>
          
          {/* Coordinate numbers */}
          <text x="237" y="27" id="coord1">875-883/029</text>
          <text x="854" y="27" id="coord2">645-380/293</text>
          <text x="167" y="136" id="coord3">SIG-582/581</text>
          <text x="430" y="312" id="coord4">OPL-517/439</text>
          <text x="372" y="429" id="coord5">APP-673/582</text>
          <text x="652" y="440" id="coord6">875-883/029</text>
          
          <animate xlinkHref="#coord4" attributeName="textContent" values="OPL-517/439;OPL-518/440;OPL-517/439" dur="5s" begin="2s" repeatCount="indefinite" />
          <animate xlinkHref="#coord6" attributeName="textContent" values="875-883/029;875-884/030;875-883/029" dur="7s" begin="1s" repeatCount="indefinite" />
        </g>
        
        {/* Status indicators */}
        <g fontFamily="monospace" fontSize="12" fill="rgba(255,255,255,0.9)">
          <text x="866" y="708">STATUS ACTIVE</text>
          <text x="631" y="708">C7</text>
          <text x="585" y="708">1</text>
          
          <circle cx="840" cy="708" r="4" fill="#00ff00">
            <animate attributeName="opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
          </circle>
        </g>
        
        {/* Central radar/tracking circle */}
        <g id="centralRadar" transform="translate(500, 360)">
          <circle cx="0" cy="0" r="100" stroke="white" strokeWidth="1" fill="none" />
          <circle cx="0" cy="0" r="100" stroke="rgba(0,255,255,0.3)" strokeWidth="1" fill="none">
            <animate attributeName="r" values="80;120;80" dur="4s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.3;0.1;0.3" dur="4s" repeatCount="indefinite" />
          </circle>
          
          {/* Radar sweep */}
          <line x1="0" y1="0" x2="0" y2="-100" stroke="rgba(0,255,255,0.6)" strokeWidth="1">
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="6s" repeatCount="indefinite" />
          </line>
          
          {/* Target triangles */}
          <use xlinkHref="#triangleMarker" x="-10" y="-75" width="20" height="20">
            <animate attributeName="y" values="-75;-72;-75" dur="3s" repeatCount="indefinite" />
          </use>
          <use xlinkHref="#triangleMarker" x="-35" y="-20" width="20" height="20" transform="scale(0.7)">
            <animate attributeName="y" values="-20;-25;-20" dur="4s" repeatCount="indefinite" />
          </use>
          <use xlinkHref="#triangleMarker" x="20" y="-50" width="20" height="20" transform="scale(0.5)">
            <animate attributeName="y" values="-50;-45;-50" dur="5s" repeatCount="indefinite" />
          </use>
        </g>
        
        {/* Marker circles */}
        <g id="markers">
          <use xlinkHref="#circleMarker" x="166" y="27" width="20" height="20" filter="url(#glow)" />
          <use xlinkHref="#circleMarker" x="817" y="27" width="20" height="20" filter="url(#glow)" />
          <use xlinkHref="#targetMarker" x="167" y="136" width="20" height="20" />
          <use xlinkHref="#circleMarker" x="332" y="165" width="20" height="20" filter="url(#glow)">
            <animate attributeName="width" values="20;24;20" dur="3s" repeatCount="indefinite" />
            <animate attributeName="height" values="20;24;20" dur="3s" repeatCount="indefinite" />
          </use>
          {[...Array(7)].map((_, i) => {
            const positions = [
              { x: 393, y: 312 },
              { x: 322, y: 429 },
              { x: 606, y: 440 },
              { x: 181, y: 508 },
              { x: 322, y: 513 },
              { x: 542, y: 624 },
              { x: 231, y: 746 }
            ];
            return (
              <use
                key={`marker${i}`}
                xlinkHref="#circleMarker"
                x={positions[i].x}
                y={positions[i].y}
                width="16"
                height="16"
              />
            );
          })}
        </g>
        
        {/* Small dot markers */}
        <g id="dots">
          {[
            { x: 170, y: 258 }, { x: 170, y: 307 }, { x: 30, y: 341 },
            { x: 170, y: 402 }, { x: 413, y: 502 }, { x: 538, y: 146 },
            { x: 960, y: 258 }, { x: 960, y: 402 }, { x: 538, y: 682 },
            { x: 688, y: 682 }, { x: 960, y: 960 }
          ].map((pos, i) => (
            <use
              key={`dot${i}`}
              xlinkHref="#dotMarker"
              x={pos.x}
              y={pos.y}
              width="10"
              height="10"
            />
          ))}
          <animate attributeName="opacity" values="1;0.4;1" dur="3s" repeatCount="indefinite" />
        </g>
        
        {/* Connection lines */}
        <g id="connectionLines" stroke="rgba(255,255,255,0.2)" strokeWidth="1">
          <line x1="176" y1="37" x2="332" y2="175" />
          <line x1="332" y1="175" x2="393" y2="312" stroke="url(#lineGradient)" />
          <line x1="322" y1="429" x2="393" y2="312" />
          <line x1="322" y1="429" x2="181" y2="508" />
          <line x1="181" y1="508" x2="322" y2="513" stroke="url(#lineGradient)" />
          <line x1="322" y1="513" x2="542" y2="624" />
          <line x1="393" y1="312" x2="606" y2="440" />
          <line x1="606" y1="440" x2="542" y2="624" stroke="url(#lineGradient)" />
        </g>
        
        {/* Random particle effects */}
        <g id="particles">
          {[...Array(6)].map((_, i) => {
            const x = 200 + i * 200;
            const y = 150 + (i % 3) * 100;
            const duration = 5 + i;
            return (
              <circle key={`particle${i}`} cx={x} cy={y} r="1" fill="white" opacity="0.6">
                <animate
                  attributeName="cy"
                  values={`${y};${y + 20};${y}`}
                  dur={`${duration}s`}
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="opacity"
                  values="0.6;0.2;0.6"
                  dur={`${duration}s`}
                  repeatCount="indefinite"
                />
              </circle>
            );
          })}
        </g>
        
        {/* Glitch effects */}
        <g id="glitchEffects">
          <rect id="glitch1" x="300" y="200" width="100" height="5" fill="rgba(0,255,255,0.3)" opacity="0">
            <animate attributeName="opacity" values="0;0.7;0" dur="0.2s" begin="3s;8s;15s;23s" />
          </rect>
          <rect id="glitch2" x="500" y="350" width="150" height="3" fill="rgba(255,255,255,0.5)" opacity="0">
            <animate attributeName="opacity" values="0;0.5;0" dur="0.1s" begin="5s;12s;18s;25s" />
          </rect>
          <rect id="glitch3" x="200" y="450" width="200" height="4" fill="rgba(255,0,0,0.2)" opacity="0">
            <animate attributeName="opacity" values="0;0.3;0" dur="0.15s" begin="7s;14s;21s;27s" />
          </rect>
        </g>
      </svg>
    </div>
  );
};

```

# src\components\memory\MemoryDashboardTile.tsx

```tsx
// src/components/memory/MemoryDashboardTile.tsx
import React, { useEffect, useRef, useState } from 'react';
import { useMemory } from '@/hooks/useMemory';
import { ConfigurationPanelItem } from '@/components/config/ConfigurationPanelItem';
import { NameValueRow } from '@/components/config/NameValueRow';
import { Button } from '@/components/button/Button';
import { createTextFlicker, createNeuralParticles, createGlitchEffect } from '@/lib/animations';
import MemoryMetrics from './MemoryMetrics';
import { useConfig } from '@/hooks/useConfig';

type MemoryDashboardTileProps = {
  accentColor: string;
};

export const MemoryDashboardTile: React.FC<MemoryDashboardTileProps> = ({
  accentColor
}) => {
  console.log("MemoryDashboardTile rendering with accentColor:", accentColor);
  
  // Get configuration settings
  const { config } = useConfig();
  console.log("MemoryDashboardTile loaded config settings:", config.settings);
  
  const { memory_enabled, memory_ws_url, memory_hpc_url } = config.settings;

  // Use memory hook with config URLs
  const { 
    connectionStatus,
    hpcStatus,
    memoryEnabled,
    setSearchText,
    search,
    clearSearch,
    selectedMemories,
    toggleSelection,
    stats,
    results: searchResults,
    processingMetrics,
    toggleMemorySystem,
    memoryWsUrl,
    memoryHpcUrl
  } = useMemory({
    defaultTensorUrl: memory_ws_url,
    defaultHpcUrl: memory_hpc_url,
    enabled: memory_enabled
  });
  
  console.log("Memory state in MemoryDashboardTile:", { 
    memory_enabled,
    memoryEnabled,
    connectionStatus,
    hpcStatus,
    memoryWsUrl,
    memoryHpcUrl  
  });
  
  const [searchQuery, setSearchQuery] = useState<string>('');
  const containerRef = useRef<HTMLDivElement>(null);
  const titleRefs = useRef<(HTMLElement | null)[]>([]);

  // Apply cyberpunk effects to the component
  useEffect(() => {
    if (containerRef.current) {
      // Add neural particles for visual effect
      const cleanupParticles = createNeuralParticles(containerRef.current, 10);
      
      // Add occasional glitch effect for cyberpunk feel
      const cleanupGlitch = createGlitchEffect(containerRef.current, 0.3);
      
      // Add text flicker effects to titles
      titleRefs.current.forEach(el => {
        if (el) createTextFlicker(el);
      });
      
      return () => {
        cleanupParticles();
        cleanupGlitch();
      };
    }
  }, []);

  // Sync memory system enabled state with config
  useEffect(() => {
    // If memory state doesn't match config, update it
    console.log(`Memory sync effect: config.enabled=${memory_enabled}, hook.enabled=${memoryEnabled}`);
    
    if (memory_enabled !== undefined && memory_enabled !== memoryEnabled) {
      console.log(`Syncing memory system state from config: ${memory_enabled}`);
      toggleMemorySystem();
    }
  }, [memory_enabled, memoryEnabled, toggleMemorySystem]);

  // Handle search submission
  const handleSearch = () => {
    if (!searchQuery.trim()) return;
    search(searchQuery);
  };

  // Handle input key press (Enter)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div 
      ref={containerRef}
      className="w-full h-full flex flex-col gap-3 overflow-y-auto scrollbar-thin scrollbar-track-gray-900/20 scrollbar-thumb-cyan-500/20"
    >
      {/* Neural Memory System Status */}
      <ConfigurationPanelItem title="Neural Memory System">
        <div className="flex flex-col gap-2">
          <NameValueRow
            name="Memory Link"
            value={
              <span className={`status-active ${connectionStatus === 'Connected' ? '' : 'text-red-400'}`}>
                {connectionStatus}
              </span>
            }
            valueColor={connectionStatus === 'Connected' ? `${accentColor}-400` : 'red-400'}
          />
          <NameValueRow
            name="HPC Engine"
            value={
              <span className={`status-active ${hpcStatus === 'Connected' ? '' : 'text-red-400'}`}>
                {hpcStatus}
              </span>
            }
            valueColor={hpcStatus === 'Connected' ? `${accentColor}-400` : 'red-400'}
          />
          <NameValueRow
            name="Memory Count"
            value={stats.memory_count}
            valueColor={`${accentColor}-400`}
          />
          <NameValueRow
            name="Neural Processing"
            value={`${stats.gpu_memory.toFixed(2)} GB`}
            valueColor={`${accentColor}-400`}
          />
          <NameValueRow
            name="Memory Status"
            value={memoryEnabled ? "Enabled" : "Disabled"}
            valueColor={memoryEnabled ? 'green-400' : 'red-400'}
          />
          <div className="flex justify-between mt-2">
            <Button
              accentColor={memoryEnabled ? 'red' : 'green'}
              onClick={toggleMemorySystem}
              className="text-xs py-1 px-3"
            >
              {memoryEnabled ? 'DISABLE MEMORY' : 'ENABLE MEMORY'}
            </Button>
            <Button
              accentColor={connectionStatus === 'Connected' ? 'red' : accentColor}
              onClick={() => {}}
              className="text-xs py-1 px-3"
            >
              {connectionStatus === 'Connected' ? 'DISCONNECT' : 'CONNECT'}
            </Button>
          </div>
        </div>
      </ConfigurationPanelItem>

      {/* Memory Search Interface */}
      <ConfigurationPanelItem title="Memory Search">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyPress}
              className={`
                w-full text-sm bg-transparent 
                text-${accentColor}-100 p-2 rounded-sm 
                focus:outline-none focus:ring-1 focus:ring-${accentColor}-700
                transition-all duration-300 backdrop-blur-sm
                bg-gray-900/20 border border-${accentColor}-800/30
              `}
              placeholder="Search neural memories..."
            />
            <Button
              accentColor={accentColor}
              onClick={handleSearch}
              className="text-xs py-1 px-3"
            >
              SEARCH
            </Button>
          </div>
        </div>
      </ConfigurationPanelItem>

      {/* Memory Results */}
      {searchResults.length > 0 && (
        <ConfigurationPanelItem title="Memory Results">
          <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
            {searchResults.map((memory) => (
              <div
                key={memory.id}
                className={`
                  relative p-3 border rounded-sm transition-all duration-300
                  ${selectedMemories.has(memory.id) 
                    ? `bg-${accentColor}-900/20 border-${accentColor}-500/50` 
                    : 'bg-gray-900/30 border-gray-800/30'
                  }
                  cursor-pointer hover:bg-${accentColor}-900/10
                `}
                onClick={() => toggleSelection(memory.id)}
              >
                {/* Left accent bar */}
                <div 
                  className={`absolute left-0 top-0 bottom-0 w-1 ${
                    selectedMemories.has(memory.id) ? `bg-${accentColor}-500` : `bg-${accentColor}-900/30`
                  }`}
                ></div>
                
                {/* Memory metrics */}
                <div className="flex flex-wrap gap-2 mb-2">
                  <div className={`text-${accentColor}-400 text-xs font-mono bg-${accentColor}-900/20 px-2 py-1 rounded-sm`}>
                    Match: {(memory.similarity * 100).toFixed(1)}%
                  </div>
                  <div className="text-amber-400 text-xs font-mono bg-amber-900/20 px-2 py-1 rounded-sm">
                    Significance: {memory.significance.toFixed(2)}
                  </div>
                  <div className="text-violet-400 text-xs font-mono bg-violet-900/20 px-2 py-1 rounded-sm">
                    Surprise: {memory.surprise.toFixed(2)}
                  </div>
                </div>
                
                {/* Memory text */}
                <div className="text-gray-300 text-sm mt-2 border-l-2 border-gray-700/50 pl-2">
                  {memory.text}
                </div>
                
                {/* Holographic overlay */}
                <div 
                  className="absolute inset-0 pointer-events-none opacity-10"
                  style={{
                    background: `linear-gradient(90deg, 
                      transparent 0%, 
                      rgba(0, 255, 255, 0.05) 25%, 
                      transparent 50%
                    )`
                  }}
                ></div>
              </div>
            ))}
          </div>
          
          {/* Actions */}
          {selectedMemories.size > 0 && (
            <div className="flex justify-between items-center mt-3 pt-2 border-t border-gray-800/30">
              <div className={`text-xs text-${accentColor}-400 font-mono`}>
                {selectedMemories.size} memories selected
              </div>
              <Button
                accentColor={accentColor}
                onClick={() => {}}
                className="text-xs py-1 px-3"
              >
                CLEAR
              </Button>
            </div>
          )}
        </ConfigurationPanelItem>
      )}

      {/* Metric Visualizations */}
      <ConfigurationPanelItem title="Neural Metrics">
        <div className="flex flex-col gap-4">
          {/* Significance & Surprise Visualization */}
          {searchResults.length > 0 && (
            <div className="w-full h-[140px] border border-gray-800/50 bg-gray-900/30 rounded-sm p-3 relative">
              <div className={`text-xs text-${accentColor}-400 mb-2 font-mono`}>Memory Significance & Surprise</div>
              
              <div className="flex h-[80px] w-full items-end justify-around relative">
                {/* Dynamic grid lines */}
                <div className="absolute inset-0 grid grid-cols-10 grid-rows-4 border-t border-l border-gray-800/30">
                  {[...Array(10)].map((_, i) => (
                    <div key={`gridcol-${i}`} className="border-r border-gray-800/20 h-full"></div>
                  ))}
                  {[...Array(4)].map((_, i) => (
                    <div key={`gridrow-${i}`} className="border-b border-gray-800/20 w-full"></div>
                  ))}
                </div>
                
                {/* Bars */}
                {searchResults.slice(0, 5).map((memory, index) => (
                  <div key={`metric-${memory.id}`} className="flex gap-1 h-full items-end z-10">
                    {/* Significance bar */}
                    <div 
                      className="w-4 bg-gradient-to-t from-amber-500 to-amber-300 relative group"
                      style={{ height: `${memory.significance * 100}%` }}
                    >
                      <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 hidden group-hover:block">
                        <div className="text-xs text-amber-400 whitespace-nowrap bg-gray-900/80 px-1 py-0.5 rounded">
                          {memory.significance.toFixed(2)}
                        </div>
                      </div>
                    </div>
                    
                    {/* Surprise bar */}
                    <div 
                      className="w-4 bg-gradient-to-t from-violet-500 to-violet-300 relative group"
                      style={{ height: `${memory.surprise * 100}%` }}
                    >
                      <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 hidden group-hover:block">
                        <div className="text-xs text-violet-400 whitespace-nowrap bg-gray-900/80 px-1 py-0.5 rounded">
                          {memory.surprise.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Legend */}
              <div className="flex justify-center gap-4 mt-2">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-amber-500"></div>
                  <span className="text-xs text-gray-400">Significance</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-violet-500"></div>
                  <span className="text-xs text-gray-400">Surprise</span>
                </div>
              </div>
              
              {/* Scan line effect */}
              <div className="absolute inset-0 scan-line pointer-events-none"></div>
            </div>
          )}
        </div>
      </ConfigurationPanelItem>

      {/* Memory System Status */}
      <div className={`flex flex-col gap-2 p-4 rounded-sm border border-${accentColor}-500/20 bg-black/30`}>
        <div className={`text-${accentColor}-400 text-sm font-mono`}>
          Memory System Status
        </div>
        <div className={`text-${accentColor}-500/70 text-xs font-mono`}>
          {connectionStatus === 'Connected' ? (
            <span className="text-green-400">● Online</span>
          ) : connectionStatus === 'Connecting' ? (
            <span className="text-yellow-400">● Connecting...</span>
          ) : (
            <span className="text-red-400">● Offline</span>
          )}
          {memoryEnabled && connectionStatus === 'Connected' ? (
            <span className="ml-2 text-green-400">| Memory Enabled</span>
          ) : connectionStatus === 'Connected' ? (
            <span className="ml-2 text-yellow-400">| Memory Disabled</span>
          ) : null}
        </div>
      </div>

      {/* Neural Activity Visualization */}
      <MemoryMetrics 
        accentColor={accentColor} 
        significance={processingMetrics?.significance}
        surprise={processingMetrics?.surprise}
      />

      {/* Connection Details */}
      <div className={`flex flex-col gap-2 p-4 rounded-sm border border-${accentColor}-500/20 bg-black/30`}>
        <div className={`text-${accentColor}-400 text-sm font-mono`}>
          Connection Details
        </div>
        <div className={`text-${accentColor}-500/70 text-xs font-mono flex flex-col gap-1`}>
          <div>Tensor Server: {memoryWsUrl}</div>
          <div>HPC Server: {memoryHpcUrl}</div>
        </div>
      </div>
    </div>
  );
};

export default MemoryDashboardTile;
```

# src\components\memory\MemoryMetrics.tsx

```tsx
// src/components/memory/MemoryMetrics.tsx
import React, { useEffect, useRef } from 'react';
import { ConfigurationPanelItem } from '@/components/config/ConfigurationPanelItem';
import { createGlitchEffect } from '@/lib/animations';

type MemoryMetricsProps = {
  accentColor: string;
  significance?: number[];
  surprise?: number[];
};

const MemoryMetrics: React.FC<MemoryMetricsProps> = ({ 
  accentColor,
  significance = [0.7, 0.5, 0.6, 0.4, 0.8],
  surprise = [0.4, 0.2, 0.7, 0.3, 0.5]
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Apply cyberpunk effects and initialize canvas
  useEffect(() => {
    if (containerRef.current) {
      const cleanupGlitch = createGlitchEffect(containerRef.current, 0.2);
      
      return () => {
        cleanupGlitch();
      };
    }
  }, []);

  // Initialize and animate the neural network visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const setCanvasSize = () => {
      if (canvas.parentElement) {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 120;
      }
    };

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    // Create neural network nodes
    const nodeCount = 20;
    const nodes = Array.from({ length: nodeCount }, (_, i) => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      radius: Math.random() * 2 + 1,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      connections: [] as number[],
      synapseStrength: i < 5 ? significance[i] : Math.random() * 0.8 + 0.2, // Use provided significance for first 5 nodes
      significance: i < 5 ? significance[i] : Math.random(),
      surprise: i < 5 ? surprise[i] : Math.random()
    }));

    // Create neural connections
    nodes.forEach((node, i) => {
      const connectionCount = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < connectionCount; j++) {
        const target = Math.floor(Math.random() * nodeCount);
        if (target !== i && !node.connections.includes(target)) {
          node.connections.push(target);
        }
      }
    });

    // Animation variables
    let pulse = 0;
    let pulseDirection = 0.01;

    // Animation function
    const animate = () => {
      if (!ctx || !canvas) return;

      // Clear canvas with semi-transparent background for trail effect
      ctx.fillStyle = 'rgba(0, 0, 31, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update pulse
      pulse += pulseDirection;
      if (pulse >= 1 || pulse <= 0) {
        pulseDirection *= -1;
      }

      // Draw connections
      nodes.forEach((node, i) => {
        node.connections.forEach(targetIndex => {
          const target = nodes[targetIndex];
          const dx = target.x - node.x;
          const dy = target.y - node.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          // Skip if too far
          if (distance > canvas.width / 3) return;

          // Calculate connection strength based on synapse strength and distance
          const strength = Math.max(0, 1 - distance / (canvas.width / 3));
          const synapseActivity = (node.synapseStrength + target.synapseStrength) / 2;
          
          // Draw connection line with color based on significance or surprise
          const significanceColor = `rgba(255, 191, 0, ${strength * synapseActivity * 0.5})`;
          const surpriseColor = `rgba(255, 0, 255, ${strength * synapseActivity * 0.5})`;
          
          // Alternate between significance and surprise colors
          const useSignificance = (i + targetIndex) % 2 === 0;
          
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(target.x, target.y);
          ctx.strokeStyle = useSignificance ? significanceColor : surpriseColor;
          ctx.lineWidth = strength * 2;
          ctx.stroke();

          // Draw pulse effect along the connection
          const pulsePosition = pulse;
          const pulsePosX = node.x + dx * pulsePosition;
          const pulsePosY = node.y + dy * pulsePosition;
          
          ctx.beginPath();
          ctx.arc(pulsePosX, pulsePosY, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = useSignificance ? 'rgba(255, 191, 0, 0.8)' : 'rgba(255, 0, 255, 0.8)';
          ctx.fill();
        });
      });

      // Update node positions
      nodes.forEach(node => {
        node.x += node.vx;
        node.y += node.vy;

        // Bounce off edges
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        
        // Color based on significance/surprise blend
        const r = Math.floor(0 + node.surprise * 255);
        const g = Math.floor(node.significance * 191);
        const b = Math.floor(node.surprise * 255);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
        ctx.fill();
      });

      requestAnimationFrame(animate);
    };

    // Start animation
    const animationId = requestAnimationFrame(animate);

    // Cleanup
    return () => {
      window.removeEventListener('resize', setCanvasSize);
      cancelAnimationFrame(animationId);
    };
  }, [significance, surprise]);

  return (
    <ConfigurationPanelItem title="Neural Activity Visualization">
      <div 
        ref={containerRef}
        className={`w-full h-[150px] border border-${accentColor}-800/30 bg-black/40 rounded-sm p-2 relative overflow-hidden`}
      >
        <canvas 
          ref={canvasRef} 
          className="w-full h-full"
        />
        
        <div className="absolute bottom-2 right-2 text-xs font-mono text-gray-500">
          Memory Synapse Activity
        </div>
        
        {/* Legend */}
        <div className="absolute top-2 right-2 flex flex-col gap-1">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-amber-500"></div>
            <span className="text-xs text-gray-400">Significance</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-fuchsia-500"></div>
            <span className="text-xs text-gray-400">Surprise</span>
          </div>
        </div>
        
        {/* Scan line effect */}
        <div className="absolute inset-0 scan-line pointer-events-none"></div>
      </div>
    </ConfigurationPanelItem>
  );
};

export default MemoryMetrics;
```

# src\components\playground\icons.tsx

```tsx
export const CheckIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="12"
    height="12"
    viewBox="0 0 12 12"
    fill="none"
  >
    <g clipPath="url(#clip0_718_9977)">
      <path
        d="M1.5 7.5L4.64706 10L10.5 2"
        stroke="white"
        strokeWidth="1.5"
        strokeLinecap="square"
      />
    </g>
    <defs>
      <clipPath id="clip0_718_9977">
        <rect width="12" height="12" fill="white" />
      </clipPath>
    </defs>
  </svg>
);

export const ChevronIcon = () => (
  <svg
    width="16"
    height="16"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className="fill-gray-200 transition-all group-hover:fill-white group-data-[state=open]:rotate-180"
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="m8 10.7.4-.3 4-4 .3-.4-.7-.7-.4.3L8 9.3 4.4 5.6 4 5.3l-.7.7.3.4 4 4 .4.3Z"
    />
  </svg>
);

```

# src\components\playground\Playground.tsx

```tsx
"use client";

import { LoadingSVG } from "@/components/button/LoadingSVG";
import { ChatMessageType } from "@/components/chat/ChatTile";
import { ColorPicker } from "@/components/colorPicker/ColorPicker";
import { AudioInputTile } from "@/components/config/AudioInputTile";
import { ConfigurationPanelItem } from "@/components/config/ConfigurationPanelItem";
import { NameValueRow } from "@/components/config/NameValueRow";
import { PlaygroundHeader } from "@/components/playground/PlaygroundHeader";
import { MemoryDashboardTile } from '@/components/memory/MemoryDashboardTile';
import {
  PlaygroundTab,
  PlaygroundTabbedTile,
  PlaygroundTile,
} from "@/components/playground/PlaygroundTile";
import { useConfig } from "@/hooks/useConfig";
import { TranscriptionTile } from "@/transcriptions/TranscriptionTile";
import {
  BarVisualizer,
  VideoTrack,
  useConnectionState,
  useDataChannel,
  useLocalParticipant,
  useRoomInfo,
  useTracks,
  useVoiceAssistant,
} from "@livekit/components-react";
import { ConnectionState, LocalParticipant, Track } from "livekit-client";
import { QRCodeSVG } from "qrcode.react";
import { ReactNode, useCallback, useEffect, useMemo, useState, useRef } from "react";
import tailwindTheme from "../../lib/tailwindTheme.preval";
import { createNeuralParticles, createGlitchEffect } from "@/lib/animations";
import { UserSettings } from "@/hooks/useConfig";

export interface PlaygroundMeta {
  name: string;
  value: string;
}

export interface PlaygroundProps {
  logo?: ReactNode;
  themeColors: string[];
  onConnect: (connect: boolean, opts?: { token: string; url: string }) => void;
}

const headerHeight = 56;

export default function Playground({
  logo,
  themeColors,
  onConnect,
}: PlaygroundProps) {
  const { config, setUserSettings } = useConfig();
  const { name } = useRoomInfo();
  const [transcripts, setTranscripts] = useState<ChatMessageType[]>([]);
  const { localParticipant } = useLocalParticipant();
  const playgroundRef = useRef<HTMLDivElement>(null);

  const voiceAssistant = useVoiceAssistant();

  const roomState = useConnectionState();
  const tracks = useTracks();

  useEffect(() => {
    if (roomState === ConnectionState.Connected) {
      localParticipant.setCameraEnabled(config.settings.inputs.camera);
      localParticipant.setMicrophoneEnabled(config.settings.inputs.mic);
    }
  }, [config, localParticipant, roomState]);

  useEffect(() => {
    if (playgroundRef.current) {
      // Add neural particle effects to the playground
      const cleanupParticles = createNeuralParticles(playgroundRef.current, 15);
      
      // Add occasional glitch effect
      const cleanupGlitch = createGlitchEffect(playgroundRef.current, 0.5);
      
      return () => {
        cleanupParticles();
        cleanupGlitch();
      };
    }
  }, []);

  const agentVideoTrack = tracks.find(
    (trackRef) =>
      trackRef.publication.kind === Track.Kind.Video &&
      trackRef.participant.isAgent
  );

  const localTracks = tracks.filter(
    ({ participant }) => participant instanceof LocalParticipant
  );
  const localVideoTrack = localTracks.find(
    ({ source }) => source === Track.Source.Camera
  );
  const localMicTrack = localTracks.find(
    ({ source }) => source === Track.Source.Microphone
  );

  const onDataReceived = useCallback(
    (msg: any) => {
      if (msg.topic === "transcription") {
        const decoded = JSON.parse(
          new TextDecoder("utf-8").decode(msg.payload)
        );
        let timestamp = new Date().getTime();
        if ("timestamp" in decoded && decoded.timestamp > 0) {
          timestamp = decoded.timestamp;
        }
        setTranscripts([
          ...transcripts,
          {
            name: "You",
            message: decoded.text,
            timestamp: timestamp,
            isSelf: true,
          },
        ]);
      }
    },
    [transcripts]
  );

  useDataChannel(onDataReceived);

  const videoTileContent = useMemo(() => {
    const videoFitClassName = `object-${config.video_fit || "cover"}`;

    const disconnectedContent = (
      <div className="flex items-center justify-center text-cyan-500 text-center w-full h-full font-mono tracking-wide opacity-70">
        <div className="glass-panel p-6 border-cyan-500/20">
          No video track. Connect to get started.
        </div>
      </div>
    );

    const loadingContent = (
      <div className="flex flex-col items-center gap-4 text-cyan-400 text-center h-full w-full font-mono tracking-wider">
        <LoadingSVG />
        <div className="digital-flicker">Initializing neural interface...</div>
      </div>
    );

    const videoContent = (
      <div className="relative w-full h-full">
        <VideoTrack
          trackRef={agentVideoTrack}
          className={`absolute top-1/2 -translate-y-1/2 ${videoFitClassName} object-position-center w-full h-full`}
        />
        
        {/* Video overlay elements for cyberpunk effect */}
        <div className="absolute inset-0 pointer-events-none border border-cyan-500/20"></div>
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent"></div>
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent"></div>
        
        {/* Status indicators */}
        <div className="absolute top-4 right-4 flex items-center gap-2">
          <div className="text-xs font-mono text-cyan-400 tracking-wider bg-black/40 px-2 py-1 rounded-sm">
            NEURAL LINK: <span className="text-cyan-300 status-active">ACTIVE</span>
          </div>
        </div>
        
        {/* Scan line effect */}
        <div className="absolute inset-0 scan-line pointer-events-none"></div>
      </div>
    );

    let content = null;
    if (roomState === ConnectionState.Disconnected) {
      content = disconnectedContent;
    } else if (agentVideoTrack) {
      content = videoContent;
    } else {
      content = loadingContent;
    }

    return (
      <div className="flex flex-col w-full grow text-cyan-500 bg-black/50 rounded-sm border border-gray-800 relative overflow-hidden">
        {content}
      </div>
    );
  }, [agentVideoTrack, config, roomState]);

  useEffect(() => {
    document.body.style.setProperty(
      "--lk-theme-color",
      // @ts-ignore
      tailwindTheme.colors[config.settings.theme_color]["500"]
    );
    document.body.style.setProperty(
      "--lk-drop-shadow",
      `var(--lk-theme-color) 0px 0px 18px`
    );
  }, [config.settings.theme_color]);

  const audioTileContent = useMemo(() => {
    const disconnectedContent = (
      <div className="flex flex-col items-center justify-center gap-2 text-cyan-500 text-center w-full font-mono tracking-wide">
        <div className="glass-panel p-6 border-cyan-500/20">
          No audio track. Connect to get started.
        </div>
      </div>
    );

    const waitingContent = (
      <div className="flex flex-col items-center gap-4 text-cyan-400 text-center w-full font-mono tracking-wider">
        <LoadingSVG />
        <div className="digital-flicker">Initializing audio processor...</div>
      </div>
    );

    const visualizerContent = (
      <div
        className={`flex items-center justify-center w-full h-48 [--lk-va-bar-width:30px] [--lk-va-bar-gap:20px] [--lk-fg:var(--lk-theme-color)] relative`}
      >
        {/* Custom visualizer wrapper for cyberpunk effect */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-full max-w-md h-full flex items-center justify-center relative border border-cyan-500/20 bg-black/30 rounded-sm p-4">
            <BarVisualizer
              state={voiceAssistant.state}
              trackRef={voiceAssistant.audioTrack}
              barCount={5}
              options={{ minHeight: 20 }}
            />
            
            {/* Neural pathway particles */}
            <div className="absolute inset-0 particles-container"></div>
            
            {/* Audio status indicator */}
            <div className="absolute bottom-2 right-2 text-xs font-mono text-cyan-400 digital-flicker tracking-wider">
              AUDIO SIGNAL: ACTIVE
            </div>
          </div>
        </div>
      </div>
    );

    if (roomState === ConnectionState.Disconnected) {
      return disconnectedContent;
    }

    if (!voiceAssistant.audioTrack) {
      return waitingContent;
    }

    return visualizerContent;
  }, [
    voiceAssistant.audioTrack,
    config.settings.theme_color,
    roomState,
    voiceAssistant.state,
  ]);

  const chatTileContent = useMemo(() => {
    if (voiceAssistant.audioTrack) {
      return (
        <TranscriptionTile
          agentAudioTrack={voiceAssistant.audioTrack}
          accentColor={config.settings.theme_color}
        />
      );
    }
    return <></>;
  }, [config.settings.theme_color, voiceAssistant.audioTrack]);

  // Update theme color setting
  const updateThemeColor = useCallback((color: string) => {
    const newSettings = { ...config.settings };
    newSettings.theme_color = color;
    setUserSettings(newSettings);
  }, [config.settings, setUserSettings]);

  const settingsTileContent = useMemo(() => {
    return (
      <div className="flex flex-col gap-3 h-full overflow-y-auto pr-2">
        {config.settings.memory_enabled && (
          <div className="w-full">
            <MemoryDashboardTile
              accentColor={config.settings.theme_color}
            />
          </div>
        )}
        {/* Debug info */}
        <div className="text-xs text-gray-400 mb-2">
          Memory enabled: {config.settings.memory_enabled !== undefined ? String(config.settings.memory_enabled) : "false"}
        </div>
        {config.description && (
          <ConfigurationPanelItem title="System Overview">
            <div className="font-mono text-cyan-300/90 text-xs leading-relaxed tracking-wide">
              {config.description}
            </div>
          </ConfigurationPanelItem>
        )}

        <ConfigurationPanelItem title="Neural Interface">
          {localParticipant && (
            <div className="flex flex-col gap-2">
              <NameValueRow
                name="Interface ID"
                value={name}
                valueColor={`${config.settings.theme_color}-400`}
              />
              <NameValueRow
                name="User Identity"
                value={localParticipant.identity}
              />
              <NameValueRow
                name="Connection Status"
                value={
                  <span className="status-active">SYNCHRONIZED</span>
                }
              />
            </div>
          )}
        </ConfigurationPanelItem>
        
        <ConfigurationPanelItem title="System Status">
          <div className="flex flex-col gap-2">
            <NameValueRow
              name="Neural Link"
              value={
                roomState === ConnectionState.Connecting ? (
                  <LoadingSVG diameter={16} strokeWidth={2} />
                ) : (
                  roomState.toUpperCase()
                )
              }
              valueColor={
                roomState === ConnectionState.Connected
                  ? `${config.settings.theme_color}-400`
                  : "gray-500"
              }
            />
            <NameValueRow
              name="AI Core"
              value={
                voiceAssistant.agent ? (
                  "ACTIVE"
                ) : roomState === ConnectionState.Connected ? (
                  <LoadingSVG diameter={12} strokeWidth={2} />
                ) : (
                  "INACTIVE"
                )
              }
              valueColor={
                voiceAssistant.agent
                  ? `${config.settings.theme_color}-400`
                  : "gray-500"
              }
            />
            <NameValueRow
              name="Quantum Database"
              value="ONLINE"
              valueColor="amber-400"
            />
          </div>
        </ConfigurationPanelItem>
        
        {localVideoTrack && (
          <ConfigurationPanelItem
            title="Visual Input"
            deviceSelectorKind="videoinput"
          >
            <div className="relative border border-cyan-500/20 rounded-sm overflow-hidden">
              <VideoTrack
                className="rounded-sm w-full cyber-loading"
                trackRef={localVideoTrack}
              />
            </div>
          </ConfigurationPanelItem>
        )}
        
        {config.show_qr && (
          <div className="w-full">
            <ConfigurationPanelItem title="Mobile Sync">
              <div className="p-2 bg-white w-fit mx-auto">
                <QRCodeSVG value={window.location.href} width="128" />
              </div>
              <div className="text-xs text-center mt-2 text-cyan-400">
                Scan to synchronize with mobile device
              </div>
            </ConfigurationPanelItem>
          </div>
        )}

        {/* Moved Audio Input up from bottom */}
        {localMicTrack && (
          <ConfigurationPanelItem
            title="Audio Input"
            deviceSelectorKind="audioinput"
          >
            <AudioInputTile trackRef={localMicTrack} />
          </ConfigurationPanelItem>
        )}
        
        {/* Moved Theme Selection to bottom */}
        <div className="w-full mt-auto">
          <ConfigurationPanelItem title="Interface Theme">
            <ColorPicker
              colors={themeColors}
              selectedColor={config.settings.theme_color}
              onSelect={updateThemeColor}
            />
          </ConfigurationPanelItem>
        </div>
      </div>
    );
  }, [
    config.description,
    config.settings,
    config.show_qr,
    localParticipant,
    name,
    roomState,
    localVideoTrack,
    localMicTrack,
    themeColors,
    updateThemeColor,
    voiceAssistant.agent,
  ]);

  let mobileTabs: PlaygroundTab[] = [];
  if (config.settings.outputs.video) {
    mobileTabs.push({
      title: "Neural Interface",
      content: (
        <PlaygroundTile
          className="w-full h-full grow"
          childrenClassName="justify-center"
        >
          {videoTileContent}
        </PlaygroundTile>
      ),
    });
  }

  if (config.settings.outputs.audio) {
    mobileTabs.push({
      title: "Audio Analysis",
      content: (
        <PlaygroundTile
          className="w-full h-full grow"
          childrenClassName="justify-center"
        >
          {audioTileContent}
        </PlaygroundTile>
      ),
    });
  }

  if (config.settings.chat) {
    mobileTabs.push({
      title: "Neural Chat",
      content: chatTileContent,
    });
  }

  mobileTabs.push({
    title: "System Controls",
    content: (
      <PlaygroundTile
        padding={false}
        backgroundColor="gray-950"
        className="h-full w-full basis-1/4 items-start overflow-y-auto flex"
        childrenClassName="h-full grow items-start"
      >
        {settingsTileContent}
      </PlaygroundTile>
    ),
  });

  return (
    <>
      <PlaygroundHeader
        title={
          <span className="text-cyan-400 tracking-wider text-glow digital-flicker font-mono">
            SYNTHIENCE.AI <span className="text-xs">v2.0</span>
          </span>
        }
        logo={logo}
        githubLink={config.github_link}
        height={headerHeight}
        accentColor={config.settings.theme_color}
        connectionState={roomState}
        onConnectClicked={() =>
          onConnect(roomState === ConnectionState.Disconnected)
        }
      />
      <div
        ref={playgroundRef}
        className={`flex gap-4 p-4 grow w-full selection:bg-${config.settings.theme_color}-900 holo-grid relative`}
        style={{ 
          height: `calc(100% - ${headerHeight}px)`,
          maxHeight: `calc(100% - ${headerHeight}px)`,
          overflow: 'hidden'
        }}
      >
        {/* Mobile layout */}
        <div className="flex flex-col gap-4 h-full w-full lg:hidden">
          <PlaygroundTabbedTile
            className="h-full w-full"
            tabs={mobileTabs}
            initialTab={0}
          />
        </div>

        {/* Desktop layout */}
        <div className="hidden lg:flex h-full w-full gap-4">
          {/* Left side: Video/Audio */}
          <div className={`flex flex-col grow basis-1/2 gap-4 h-full ${
            !config.settings.outputs.audio && !config.settings.outputs.video
              ? "hidden"
              : "flex"
          }`}>
            {config.settings.outputs.video && (
              <PlaygroundTile
                title="Neural Interface"
                className="w-full flex-1"
                childrenClassName="justify-center"
              >
                {videoTileContent}
              </PlaygroundTile>
            )}
            {config.settings.outputs.audio && (
              <PlaygroundTile
                title="Audio Analysis"
                className="w-full flex-1"
                childrenClassName="justify-center"
              >
                {audioTileContent}
              </PlaygroundTile>
            )}
          </div>

          {/* Middle: Chat */}
          {config.settings.chat && (
            <PlaygroundTile
              title="Neural Chat"
              className="h-full grow basis-1/4"
            >
              {chatTileContent}
            </PlaygroundTile>
          )}

          {/* Right side: Settings/Controls */}
          <PlaygroundTile
            padding={false}
            backgroundColor="gray-950"
            className="h-full w-full basis-1/4 max-w-[400px] overflow-hidden"
            childrenClassName="h-full"
          >
            {settingsTileContent}
          </PlaygroundTile>
        </div>
      </div>
    </>
  );
}
```

# src\components\playground\PlaygroundDeviceSelector.tsx

```tsx
import { useMediaDeviceSelect } from "@livekit/components-react";
import { useEffect, useState } from "react";

type PlaygroundDeviceSelectorProps = {
  kind: MediaDeviceKind;
};

export const PlaygroundDeviceSelector = ({
  kind,
}: PlaygroundDeviceSelectorProps) => {
  const [showMenu, setShowMenu] = useState(false);
  const deviceSelect = useMediaDeviceSelect({ kind: kind });
  const [selectedDeviceName, setSelectedDeviceName] = useState("");

  useEffect(() => {
    deviceSelect.devices.forEach((device) => {
      if (device.deviceId === deviceSelect.activeDeviceId) {
        setSelectedDeviceName(device.label);
      }
    });
  }, [deviceSelect.activeDeviceId, deviceSelect.devices, selectedDeviceName]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showMenu) {
        setShowMenu(false);
      }
    };
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [showMenu]);

  return (
    <div className="relative z-20">
      <button
        className="flex gap-2 items-center px-2 py-1 bg-gray-900/60 text-cyan-400 border border-gray-800 
        rounded-sm hover:bg-gray-800/80 hover:border-cyan-500/30 transition-all duration-300"
        onClick={(e) => {
          setShowMenu(!showMenu);
          e.stopPropagation();
        }}
      >
        <span className="max-w-[80px] overflow-ellipsis overflow-hidden whitespace-nowrap font-mono text-xs tracking-wide">
          {selectedDeviceName || "Select Device"}
        </span>
        <ChevronSVG />
      </button>
      
      <div
        className="absolute right-0 top-8 bg-gray-900/95 text-gray-300 border border-cyan-500/20 
        rounded-sm z-10 w-48 backdrop-blur-sm overflow-hidden shadow-lg shadow-cyan-500/10"
        style={{
          display: showMenu ? "block" : "none",
        }}
      >
        {deviceSelect.devices.map((device, index) => {
          return (
            <div
              onClick={() => {
                deviceSelect.setActiveMediaDevice(device.deviceId);
                setShowMenu(false);
              }}
              className={`
                ${device.deviceId === deviceSelect.activeDeviceId ? "text-cyan-400 bg-gray-800/70" : "text-gray-500 bg-gray-900/80"} 
                text-xs py-2 px-3 cursor-pointer hover:bg-gray-800 hover:text-cyan-300 transition-colors
                border-b border-gray-800/50 last:border-b-0 font-mono
              `}
              key={index}
            >
              {device.label}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const ChevronSVG = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="12"
    height="12"
    viewBox="0 0 16 16"
    fill="none"
    className="opacity-70"
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M3 5H5V7H3V5ZM7 9V7H5V9H7ZM9 9V11H7V9H9ZM11 7V9H9V7H11ZM11 7V5H13V7H11Z"
      fill="currentColor"
      fillOpacity="0.8"
    />
  </svg>
);
```

# src\components\playground\PlaygroundHeader.tsx

```tsx
import { Button } from "@/components/button/Button";
import { LoadingSVG } from "@/components/button/LoadingSVG";
import { SettingsDropdown } from "@/components/playground/SettingsDropdown";
import { useConfig } from "@/hooks/useConfig";
import { ConnectionState } from "livekit-client";
import { ReactNode, useEffect, useRef } from "react";
import { createTextFlicker } from "@/lib/animations";

type PlaygroundHeaderProps = {
  logo?: ReactNode;
  title?: ReactNode;
  githubLink?: string;
  height: number;
  accentColor: string;
  connectionState: ConnectionState;
  onConnectClicked: () => void;
};

export const PlaygroundHeader = ({
  logo,
  title,
  githubLink,
  accentColor,
  height,
  onConnectClicked,
  connectionState,
}: PlaygroundHeaderProps) => {
  const { config } = useConfig();
  const titleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (titleRef.current) {
      // Add text flicker effect to the title
      createTextFlicker(titleRef.current);
    }
  }, []);

  return (
    <div
      className={`
        flex gap-4 pt-4 text-${accentColor}-500 justify-between items-center shrink-0
        glass-panel border-b border-${accentColor}-800/30 backdrop-blur-md
      `}
      style={{
        height: height + "px",
      }}
    >
      <div className="flex items-center gap-3 basis-2/3">
        <div className="flex lg:basis-1/2">
          <a href="https://livekit.io" className="hover:opacity-80 transition-opacity">
            {logo ?? <SynthienceLogo accentColor={accentColor} />}
          </a>
        </div>
        
        <div 
          ref={titleRef}
          className="lg:basis-1/2 lg:text-center text-xs lg:text-base lg:font-medium text-cyan-400 digital-flicker tracking-wider"
        >
          {title}
        </div>
      </div>
      
      <div className="flex basis-1/3 justify-end items-center gap-3">
        {githubLink && (
          <a
            href={githubLink}
            target="_blank"
            className={`text-white hover:text-${accentColor}-300 transition-colors duration-300`}
            title="View on GitHub"
          >
            <GithubSVG />
          </a>
        )}
        
        {config.settings.editable && <SettingsDropdown />}
        
        <Button
          accentColor={
            connectionState === ConnectionState.Connected ? "red" : accentColor
          }
          disabled={connectionState === ConnectionState.Connecting}
          onClick={() => {
            onConnectClicked();
          }}
        >
          {connectionState === ConnectionState.Connecting ? (
            <LoadingSVG />
          ) : connectionState === ConnectionState.Connected ? (
            "Disconnect"
          ) : (
            "Initialize"
          )}
        </Button>
      </div>
    </div>
  );
};

const SynthienceLogo = ({ accentColor }: { accentColor: string }) => (
  <svg
    width="32"
    height="32"
    viewBox="0 0 32 32"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={`text-${accentColor}-500`}
  >
    <path
      d="M16 0C7.163 0 0 7.163 0 16C0 24.837 7.163 32 16 32C24.837 32 32 24.837 32 16C32 7.163 24.837 0 16 0ZM16 2C23.732 2 30 8.268 30 16C30 23.732 23.732 30 16 30C8.268 30 2 23.732 2 16C2 8.268 8.268 2 16 2Z"
      fill="currentColor"
    />
    <path
      d="M16 6C10.477 6 6 10.477 6 16C6 21.523 10.477 26 16 26C21.523 26 26 21.523 26 16C26 10.477 21.523 6 16 6ZM16 8C20.418 8 24 11.582 24 16C24 20.418 20.418 24 16 24C11.582 24 8 20.418 8 16C8 11.582 11.582 8 16 8Z"
      fill="currentColor"
    />
    <path
      d="M16 12C13.791 12 12 13.791 12 16C12 18.209 13.791 20 16 20C18.209 20 20 18.209 20 16C20 13.791 18.209 12 16 12Z"
      fill="currentColor"
    />
    <path
      d="M24 2H30V8"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    />
    <path
      d="M8 30H2V24"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    />
  </svg>
);

const GithubSVG = () => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 98 96"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z"
      fill="currentColor"
    />
  </svg>
);
```

# src\components\playground\PlaygroundTile.tsx

```tsx
import { ReactNode, useState, useEffect, useRef } from "react";
import { createNeuralParticles } from "@/lib/animations";

const titleHeight = 32;

type PlaygroundTileProps = {
  title?: string;
  children?: ReactNode;
  className?: string;
  childrenClassName?: string;
  padding?: boolean;
  backgroundColor?: string;
};

export type PlaygroundTab = {
  title: string;
  content: ReactNode;
};

export type PlaygroundTabbedTileProps = {
  tabs: PlaygroundTab[];
  initialTab?: number;
} & PlaygroundTileProps;

export const PlaygroundTile: React.FC<PlaygroundTileProps> = ({
  children,
  title,
  className,
  childrenClassName,
  padding = true,
  backgroundColor = "transparent",
}) => {
  const contentPadding = padding ? 4 : 0;
  const tileRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (tileRef.current) {
      // Add neural particles for visual effect (low count for better performance)
      const cleanup = createNeuralParticles(tileRef.current, 5);
      return cleanup;
    }
  }, []);
  
  return (
    <div
      ref={tileRef}
      className={`
        flex flex-col relative glass-panel
        text-gray-300 bg-${backgroundColor}
        overflow-hidden
        transition-all duration-300
        hover:shadow-lg hover:shadow-cyan-500/10
        ${className || ""}
      `}
    >
      {title && (
        <div
          className="flex items-center justify-between text-xs uppercase py-2 px-4 border-b border-b-gray-800 tracking-wider text-cyan-400 text-glow"
          style={{
            height: `${titleHeight}px`,
            background: "rgba(0, 15, 30, 0.6)",
          }}
        >
          <h2 className="digital-flicker">{title}</h2>
          
          {/* Decorative elements for the cyberpunk HUD look */}
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-cyan-500/70 animate-pulse"></div>
            <div className="text-xs text-cyan-500/80 font-mono">SYN.ACTIVE</div>
          </div>
        </div>
      )}
      
      <div
        className={`
          flex flex-col items-center grow w-full 
          relative z-10
          ${childrenClassName || ""}
        `}
        style={{
          height: `calc(100% - ${title ? titleHeight + "px" : "0px"})`,
          padding: `${contentPadding * 4}px`,
        }}
      >
        {children}
      </div>
    </div>
  );
};

export const PlaygroundTabbedTile: React.FC<PlaygroundTabbedTileProps> = ({
  tabs,
  initialTab = 0,
  className,
  childrenClassName,
  backgroundColor = "transparent",
}) => {
  const contentPadding = 4;
  const [activeTab, setActiveTab] = useState(initialTab);
  const tileRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (tileRef.current) {
      // Add neural particles for visual effect
      const cleanup = createNeuralParticles(tileRef.current, 8);
      return cleanup;
    }
  }, []);
  
  if (activeTab >= tabs.length) {
    return null;
  }
  
  return (
    <div
      ref={tileRef}
      className={`
        flex flex-col h-full glass-panel 
        text-gray-400 bg-${backgroundColor}
        overflow-hidden
        transition-all duration-300
        ${className || ""}
      `}
    >
      <div
        className="flex items-center justify-start text-xs uppercase border-b border-b-gray-800 tracking-wider"
        style={{
          height: `${titleHeight}px`,
          background: "rgba(0, 15, 30, 0.6)",
        }}
      >
        {tabs.map((tab, index) => (
          <button
            key={index}
            className={`
              px-4 py-2 rounded-sm
              border-r border-r-gray-800
              transition-all duration-300
              ${
                index === activeTab
                  ? "bg-cyan-900/20 text-cyan-400 text-glow border-b-2 border-b-cyan-500"
                  : "bg-transparent text-gray-500 hover:text-gray-300 hover:bg-gray-800/30"
              }
            `}
            onClick={() => setActiveTab(index)}
          >
            {tab.title}
          </button>
        ))}
      </div>
      
      <div
        className={`w-full relative ${childrenClassName || ""}`}
        style={{
          height: `calc(100% - ${titleHeight}px)`,
          padding: `${contentPadding * 4}px`,
        }}
      >
        {tabs[activeTab].content}
      </div>
    </div>
  );
};
```

# src\components\playground\SettingsDropdown.tsx

```tsx
// src/components/playground/SettingsDropdown.tsx
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useConfig } from "@/hooks/useConfig";
import React, { useState } from "react";

// Extend the base UserSettings type from useConfig
import type { UserSettings as BaseUserSettings } from "@/hooks/useConfig";

export interface UserSettings extends BaseUserSettings {
  memory_enabled: boolean;
  memory_ws_url: string;
  memory_hpc_url: string;
}

type SettingType = "chat" | "memory" | "inputs" | "outputs" | "separator";

type Setting = {
  title: string;
  type: SettingType;
  key: keyof UserSettings | "camera" | "mic" | "video" | "audio" | "separator_1" | "separator_2" | "separator_3";
};

const settings: Setting[] = [
  {
    title: "Show chat",
    type: "chat",
    key: "chat",
  },
  {
    title: "---",
    type: "separator",
    key: "separator_1",
  },
  {
    title: "Camera",
    type: "inputs",
    key: "camera",
  },
  {
    title: "Microphone",
    type: "inputs",
    key: "mic",
  },
  {
    title: "---",
    type: "separator",
    key: "separator_2",
  },
  {
    title: "Video output",
    type: "outputs",
    key: "video",
  },
  {
    title: "Audio output",
    type: "outputs",
    key: "audio",
  },
  {
    title: "---",
    type: "separator",
    key: "separator_3",
  },
  {
    title: "Memory System",
    type: "memory",
    key: "memory_enabled",
  }
];

// SVG icons as React components
const CheckIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="12"
    height="12"
    viewBox="0 0 12 12"
    fill="none"
  >
    <g clipPath="url(#clip0_718_9977)">
      <path
        d="M1.5 7.5L4.64706 10L10.5 2"
        stroke="white"
        strokeWidth="1.5"
        strokeLinecap="square"
      />
    </g>
    <defs>
      <clipPath id="clip0_718_9977">
        <rect width="12" height="12" fill="white" />
      </clipPath>
    </defs>
  </svg>
);

const ChevronIcon: React.FC = () => (
  <svg
    width="16" 
    height="16"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className="w-3 h-3 fill-gray-200 transition-all group-hover:fill-white"
    style={{ transform: "rotate(0deg)", transition: "transform 0.2s ease" }}
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="m8 10.7.4-.3 4-4 .3-.4-.7-.7-.4.3L8 9.3 4.4 5.6 4 5.3l-.7.7.3.4 4 4 .4.3Z"
    />
  </svg>
);

export const SettingsDropdown = () => {
  const { config, setUserSettings } = useConfig();
  const [isOpen, setIsOpen] = useState(false);

  const getValue = (setting: Setting): boolean | string | undefined => {
    if (setting.type === "separator") return undefined;
    
    if (setting.type === "chat") {
      return config.settings.chat;
    }
    if (setting.type === "memory") {
      const key = setting.key as "memory_enabled" | "memory_ws_url" | "memory_hpc_url";
      return (config.settings as UserSettings)[key];
    }
    if (setting.type === "inputs") {
      const key = setting.key as "camera" | "mic";
      return config.settings.inputs[key];
    } 
    if (setting.type === "outputs") {
      const key = setting.key as "video" | "audio";
      return config.settings.outputs[key];
    }
    return undefined;
  };

  const handleChange = (setting: Setting, newValue: boolean | string) => {
    if (setting.type === "separator") return;
    
    const newSettings = { ...config.settings } as UserSettings;
    console.log("handleChange called with setting:", setting, "and new value:", newValue);
    if (setting.type === "chat") {
      newSettings.chat = newValue as boolean;
    } else if (setting.type === "memory") {
      const key = setting.key as "memory_enabled" | "memory_ws_url" | "memory_hpc_url";
      if (key === "memory_enabled") {
        newSettings[key] = newValue as boolean;
        console.log(`Memory system ${newValue ? 'enabled' : 'disabled'} in settings`, newSettings);
        
        // Update memory service state if the component is loaded
        try {
          const { getMemoryService } = require('@/lib/memoryService');
          const memoryService = getMemoryService();
          memoryService.setEnabled(newValue as boolean);
          console.log("Successfully updated memory service directly");
        } catch (e) {
          console.warn('Could not update memory service directly:', e);
        }
      } else if (key === "memory_ws_url" || key === "memory_hpc_url") {
        newSettings[key] = newValue as string;
      }
    } else if (setting.type === "inputs") {
      const key = setting.key as "camera" | "mic";
      newSettings.inputs[key] = newValue as boolean;
    } else if (setting.type === "outputs") {
      const key = setting.key as "video" | "audio";
      newSettings.outputs[key] = newValue as boolean;
    }
    console.log("New settings:", newSettings);
    setUserSettings(newSettings);
  };

  return (
    <DropdownMenu.Root modal={false} onOpenChange={setIsOpen}>
      <DropdownMenu.Trigger asChild>
        <button className="group inline-flex items-center gap-1 rounded-md py-1 px-2 text-gray-300 hover:bg-gray-800 hover:text-gray-100 transition-colors">
          <span className="text-sm">Settings</span>
          <svg
            width="16" 
            height="16"
            fill="currentColor"
            xmlns="http://www.w3.org/2000/svg"
            className="w-3 h-3 transition-transform duration-200"
            style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0)' }}
          >
            <path
              fillRule="evenodd"
              clipRule="evenodd"
              d="m8 10.7.4-.3 4-4 .3-.4-.7-.7-.4.3L8 9.3 4.4 5.6 4 5.3l-.7.7.3.4 4 4 .4.3Z"
            />
          </svg>
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          sideOffset={5}
          align="start"
          collisionPadding={16}
          className="z-50 animate-fadeIn"
        >
          <div 
            style={{ minWidth: "240px" }} 
            className="bg-gray-900 rounded-md overflow-hidden shadow-xl border border-gray-800 backdrop-blur-sm"
          >
            <div className="py-1 px-2 text-xs uppercase tracking-wider text-cyan-400 border-b border-gray-800">
              System Settings
            </div>
            
            {settings.map((setting) => {
              if (setting.type === "separator") {
                return (
                  <div
                    key={setting.key}
                    className="h-[1px] bg-gray-800 mx-3 my-1"
                  />
                );
              }

              const value = getValue(setting);
              return (
                <DropdownMenu.Item
                  key={setting.key}
                  onSelect={(e) => {
                    e?.preventDefault();
                    if (typeof value === 'boolean') {
                      handleChange(setting, !value);
                    }
                  }}
                  className="flex items-center gap-3 px-3 py-2 text-sm hover:bg-gray-800 cursor-pointer outline-none text-gray-300 hover:text-cyan-300 transition-colors"
                >
                  <div className={`w-5 h-5 flex items-center justify-center border rounded-sm ${value ? 'bg-cyan-900/30 border-cyan-500/50' : 'bg-gray-900/50 border-gray-700'}`}>
                    {value && <CheckIcon />}
                  </div>
                  <span>{setting.title}</span>
                </DropdownMenu.Item>
              );
            })}
            
            {/* Advanced settings section */}
            {(config.settings as UserSettings).memory_enabled && (
              <>
                <div className="h-[1px] bg-gray-800 mx-3 my-1" />
                <div className="py-1 px-2 text-xs uppercase tracking-wider text-cyan-400 border-b border-gray-800">
                  Memory Settings
                </div>
                <div className="p-3 text-xs text-gray-400">
                  Memory system is enabled. Configure connection settings in the Memory panel.
                </div>
              </>
            )}
            
            <div className="p-2 flex justify-end border-t border-gray-800">
              <a 
                href="https://docs.livekit.io/agents" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-xs text-cyan-500 hover:text-cyan-400 transition-colors"
              >
                Learn more
              </a>
            </div>
          </div>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
};
```

# src\components\PlaygroundConnect.tsx

```tsx
import { useConfig } from "@/hooks/useConfig";
import { CLOUD_ENABLED, CloudConnect } from "../cloud/CloudConnect";
import { Button } from "./button/Button";
import { useState, useEffect, useRef } from "react";
import { ConnectionMode } from "@/hooks/useConnection";
import { createGlitchEffect, createNeuralParticles } from "@/lib/animations";

type PlaygroundConnectProps = {
  accentColor: string;
  onConnectClicked: (mode: ConnectionMode) => void;
};

const ConnectTab = ({ active, onClick, children }: any) => {
  let className = "px-4 py-2 text-sm tracking-wide uppercase font-mono transition-all duration-300";

  if (active) {
    className += " border-b-2 border-cyan-500 text-cyan-400 text-glow";
  } else {
    className += " text-gray-500 border-b border-transparent hover:text-gray-300";
  }

  return (
    <button className={className} onClick={onClick}>
      {children}
    </button>
  );
};

const TokenConnect = ({
  accentColor,
  onConnectClicked,
}: PlaygroundConnectProps) => {
  const { setUserSettings, config } = useConfig();
  const [url, setUrl] = useState(config.settings.ws_url);
  const [token, setToken] = useState(config.settings.token);
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (containerRef.current) {
      // Add neural particles for visual effect
      const cleanupParticles = createNeuralParticles(containerRef.current, 10);
      // Add glitch effect for cyberpunk feel
      const cleanupGlitch = createGlitchEffect(containerRef.current, 0.5);
      
      return () => {
        cleanupParticles();
        cleanupGlitch();
      };
    }
  }, []);

  return (
    <div 
      ref={containerRef}
      className="flex flex-col gap-6 p-8 bg-gray-950/80 w-full text-white border-t border-cyan-900/30 glass-panel relative overflow-hidden"
    >
      <div className="flex flex-col gap-4 relative z-10">
        <div className="text-xs text-cyan-400 uppercase tracking-wider mb-2 digital-flicker">Enter Neural Link Parameters</div>
        
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="text-cyan-300 text-sm bg-black/50 border border-cyan-800/30 rounded-sm px-4 py-3 
            focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/30
            font-mono placeholder-cyan-800/50 transition-all duration-300"
          placeholder="wss://neural.link.url"
        />
        
        <textarea
          value={token}
          onChange={(e) => setToken(e.target.value)}
          className="text-cyan-300 text-sm bg-black/50 border border-cyan-800/30 rounded-sm px-4 py-3 
            focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/30 
            font-mono placeholder-cyan-800/50 min-h-[100px] transition-all duration-300"
          placeholder="Neural link authentication token..."
        />
      </div>
      
      <Button
        accentColor={accentColor}
        className="w-full py-3 text-lg tracking-wider"
        onClick={() => {
          const newSettings = { ...config.settings };
          newSettings.ws_url = url;
          newSettings.token = token;
          setUserSettings(newSettings);
          onConnectClicked("manual");
        }}
      >
        Initialize Neural Link
      </Button>
      
      <a
        href="https://kitt.livekit.io/"
        className={`text-xs text-${accentColor}-500 hover:text-${accentColor}-400 text-center transition-colors duration-300 underline`}
      >
        Don't have credentials? Try the KITT demo environment.
      </a>
      
      {/* Scan line effect */}
      <div className="absolute inset-0 scan-line"></div>
    </div>
  );
};

export const PlaygroundConnect = ({
  accentColor,
  onConnectClicked,
}: PlaygroundConnectProps) => {
  const [showCloud, setShowCloud] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (containerRef.current) {
      // Add neural particles for visual effect
      const cleanupParticles = createNeuralParticles(containerRef.current, 15);
      
      return () => {
        cleanupParticles();
      };
    }
  }, []);
  
  const copy = CLOUD_ENABLED
    ? "Initialize neural interface with LiveKit Cloud or manually with access credentials"
    : "Initialize neural interface with your access credentials";
    
  return (
    <div 
      ref={containerRef}
      className="flex left-0 top-0 w-full h-full bg-black/90 items-center justify-center text-center gap-2 relative overflow-hidden"
    >
      {/* Holographic grid overlay */}
      <div className="absolute inset-0 pointer-events-none holo-grid"></div>
      
      <div className="min-h-[540px] relative z-10">
        <div className="flex flex-col bg-gray-950/80 w-full max-w-[520px] rounded-md text-white border border-cyan-900/30 glass-panel overflow-hidden">
          <div className="flex flex-col gap-2">
            <div className="px-10 space-y-4 py-8">
              <h1 className="text-2xl text-cyan-400 tracking-wider font-mono digital-flicker">
                SYNTHIENCE.AI
              </h1>
              <p className="text-sm text-gray-400 tracking-wide">
                {copy}
              </p>
            </div>
            
            {CLOUD_ENABLED && (
              <div className="flex justify-center pt-2 gap-4 border-b border-t border-gray-900/70">
                <ConnectTab
                  active={showCloud}
                  onClick={() => {
                    setShowCloud(true);
                  }}
                >
                  Cloud Access
                </ConnectTab>
                <ConnectTab
                  active={!showCloud}
                  onClick={() => {
                    setShowCloud(false);
                  }}
                >
                  Manual Access
                </ConnectTab>
              </div>
            )}
          </div>
          
          <div className="flex flex-col bg-gray-900/30 flex-grow">
            {showCloud && CLOUD_ENABLED ? (
              <CloudConnect accentColor={accentColor} />
            ) : (
              <TokenConnect
                accentColor={accentColor}
                onConnectClicked={onConnectClicked}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
```

# src\components\toast\PlaygroundToast.tsx

```tsx
import { useToast } from "./ToasterProvider";

export type ToastType = "error" | "success" | "info";
export type ToastProps = {
  message: string;
  type: ToastType;
  onDismiss: () => void;
};

export const PlaygroundToast = () => {
  const { toastMessage, setToastMessage } = useToast();
  const color =
    toastMessage?.type === "error"
      ? "red"
      : toastMessage?.type === "success"
      ? "green"
      : "amber";

  return (
    <div
      className={`absolute text-sm break-words px-4 pr-12 py-2 bg-${color}-950 rounded-sm border border-${color}-800 text-${color}-400 top-4 left-4 right-4`}
    >
      <button
        className={`absolute right-2 border border-transparent rounded-md px-2 hover:bg-${color}-900 hover:text-${color}-300`}
        onClick={() => {
          setToastMessage(null);
        }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
        >
          <path
            fillRule="evenodd"
            clipRule="evenodd"
            d="M5.29289 5.29289C5.68342 4.90237 6.31658 4.90237 6.70711 5.29289L12 10.5858L17.2929 5.29289C17.6834 4.90237 18.3166 4.90237 18.7071 5.29289C19.0976 5.68342 19.0976 6.31658 18.7071 6.70711L13.4142 12L18.7071 17.2929C19.0976 17.6834 19.0976 18.3166 18.7071 18.7071C18.3166 19.0976 17.6834 19.0976 17.2929 18.7071L12 13.4142L6.70711 18.7071C6.31658 19.0976 5.68342 19.0976 5.29289 18.7071C4.90237 18.3166 4.90237 17.6834 5.29289 17.2929L10.5858 12L5.29289 6.70711C4.90237 6.31658 4.90237 5.68342 5.29289 5.29289Z"
            fill="currentColor"
          />
        </svg>
      </button>
      {toastMessage?.message}
    </div>
  );
};

```

# src\components\toast\ToasterProvider.tsx

```tsx
"use client"

import React, { createContext, useState } from "react";
import { ToastType } from "./PlaygroundToast";

type ToastProviderData = {
  setToastMessage: (
    message: { message: string; type: ToastType } | null
  ) => void;
  toastMessage: { message: string; type: ToastType } | null;
};

const ToastContext = createContext<ToastProviderData | undefined>(undefined);

export const ToastProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [toastMessage, setToastMessage] = useState<{message: string, type: ToastType} | null>(null);

  return (
    <ToastContext.Provider
      value={{
        toastMessage,
        setToastMessage
      }}
    >
      {children}
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const context = React.useContext(ToastContext);
  if (context === undefined) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}
```

# src\hooks\backup\useConfig.ts

```ts
"use client";

import { getCookie, setCookie } from "cookies-next";
import jsYaml from "js-yaml";
import { useRouter } from "next/navigation";
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type UserSettings = {
  editable: boolean;
  theme_color: string;
  chat: boolean;
  inputs: {
    camera: boolean;
    mic: boolean;
  };
  outputs: {
    audio: boolean;
    video: boolean;
  };
  ws_url: string;
  token: string;
  memory_enabled: boolean;
  memory_ws_url: string;
  memory_hpc_url: string;
};

export type AppConfig = {
  title: string;
  description: string;
  github_link?: string;
  video_fit?: "cover" | "contain";
  settings: UserSettings;
  show_qr?: boolean;
};

// Default config
const defaultConfig: AppConfig = {
  title: "SYNTHIENCE Neural Playground",
  description: "A lucid neural presence—real-time, memory-driven, and always aware.",
  video_fit: "cover",
  settings: {
    editable: true,
    theme_color: "cyan",
    chat: true,
    inputs: {
      camera: true,
      mic: true,
    },
    outputs: {
      audio: true,
      video: true,
    },
    ws_url: "",
    token: "",
    memory_enabled: false,
    memory_ws_url: "ws://localhost:5001",
    memory_hpc_url: "ws://localhost:5005",
  },
  show_qr: false,
};

// Convert boolean to "1" or "0" string
const boolToString = (b: boolean): string => (b ? "1" : "0");

// Hook to get app config
const useAppConfig = (): AppConfig => {
  return useMemo(() => {
    if (process.env.NEXT_PUBLIC_APP_CONFIG) {
      try {
        const parsedConfig = jsYaml.load(process.env.NEXT_PUBLIC_APP_CONFIG) as AppConfig;
        
        // Set defaults for missing values
        if (!parsedConfig.settings) {
          parsedConfig.settings = { ...defaultConfig.settings };
        }
        
        if (parsedConfig.settings.editable === undefined) {
          parsedConfig.settings.editable = true;
        }
        
        if (parsedConfig.settings.memory_enabled === undefined) {
          parsedConfig.settings.memory_enabled = false;
        }
        
        if (parsedConfig.settings.memory_ws_url === undefined) {
          parsedConfig.settings.memory_ws_url = "ws://localhost:5001";
        }
        
        if (parsedConfig.settings.memory_hpc_url === undefined) {
          parsedConfig.settings.memory_hpc_url = "ws://localhost:5005";
        }
        
        if (parsedConfig.description === undefined) {
          parsedConfig.description = defaultConfig.description;
        }
        
        if (parsedConfig.video_fit === undefined) {
          parsedConfig.video_fit = "cover";
        }
        
        if (parsedConfig.show_qr === undefined) {
          parsedConfig.show_qr = false;
        }
        
        return parsedConfig;
      } catch (e) {
        console.error("Error parsing app config:", e);
      }
    }
    return { ...defaultConfig };
  }, []);
};

// Config context type
interface ConfigContextType {
  config: AppConfig;
  setUserSettings: (settings: UserSettings) => void;
}

// Create context
const ConfigContext = createContext<ConfigContextType | undefined>(undefined);

// Provider component
export const ConfigProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const appConfig = useAppConfig();
  const router = useRouter();
  const [localColorOverride, setLocalColorOverride] = useState<string | null>(null);
  const [config, setConfig] = useState<AppConfig>(defaultConfig);
  
  // Get settings from URL parameters
  const getSettingsFromUrl = useCallback(() => {
    if (typeof window === "undefined" || !window.location.hash) {
      return null;
    }
    
    if (!appConfig.settings.editable) {
      return null;
    }
    
    const params = new URLSearchParams(window.location.hash.replace("#", ""));
    
    return {
      editable: true,
      theme_color: params.get("theme_color") || defaultConfig.settings.theme_color,
      chat: params.get("chat") === "1",
      inputs: {
        camera: params.get("cam") === "1",
        mic: params.get("mic") === "1",
      },
      outputs: {
        audio: params.get("audio") === "1",
        video: params.get("video") === "1",
      },
      ws_url: "",
      token: "",
      memory_enabled: params.get("memory") === "1",
      memory_ws_url: params.get("memory_ws") || "ws://localhost:5001",
      memory_hpc_url: params.get("memory_hpc") || "ws://localhost:5005",
    } as UserSettings;
  }, [appConfig.settings.editable]);
  
  // Get settings from cookies
  const getSettingsFromCookies = useCallback(() => {
    if (!appConfig.settings.editable) {
      return null;
    }
    
    const jsonSettings = getCookie("lk_settings");
    if (!jsonSettings) {
      return null;
    }
    
    try {
      return JSON.parse(jsonSettings as string) as UserSettings;
    } catch (e) {
      console.error("Error parsing settings from cookies:", e);
      return null;
    }
  }, [appConfig.settings.editable]);
  
  // Save settings to URL
  const setUrlSettings = useCallback((settings: UserSettings) => {
    const params = new URLSearchParams({
      cam: boolToString(settings.inputs.camera),
      mic: boolToString(settings.inputs.mic),
      video: boolToString(settings.outputs.video),
      audio: boolToString(settings.outputs.audio),
      chat: boolToString(settings.chat),
      theme_color: settings.theme_color || "cyan",
      memory: boolToString(settings.memory_enabled),
    });
    
    // Add memory URLs if memory is enabled
    if (settings.memory_enabled) {
      params.set("memory_ws", settings.memory_ws_url);
      params.set("memory_hpc", settings.memory_hpc_url);
    }
    
    router.replace("/#" + params.toString());
  }, [router]);
  
  // Save settings to cookies
  const setCookieSettings = useCallback((settings: UserSettings) => {
    try {
      const json = JSON.stringify(settings);
      setCookie("lk_settings", json);
    } catch (e) {
      console.error("Error saving settings to cookies:", e);
    }
  }, []);
  
  // Get config with settings from URL or cookies
  const getConfig = useCallback(() => {
    const result = { ...appConfig };
    
    // If settings are not editable, just set color override if any
    if (!result.settings.editable) {
      if (localColorOverride) {
        result.settings.theme_color = localColorOverride;
      }
      return result;
    }
    
    // Try to get settings from cookies or URL
    const cookieSettings = getSettingsFromCookies();
    const urlSettings = getSettingsFromUrl();
    
    // Sync settings between cookies and URL
    if (!cookieSettings && urlSettings) {
      setCookieSettings(urlSettings);
    }
    
    if (!urlSettings && cookieSettings) {
      setUrlSettings(cookieSettings);
    }
    
    // Get updated cookie settings
    const newSettings = getSettingsFromCookies();
    if (newSettings) {
      result.settings = newSettings;
    }
    
    return result;
  }, [
    appConfig,
    localColorOverride,
    getSettingsFromCookies,
    getSettingsFromUrl,
    setCookieSettings,
    setUrlSettings,
  ]);
  
  // Update user settings
  const setUserSettings = useCallback((settings: UserSettings) => {
    // If settings are not editable, just update color
    if (!appConfig.settings.editable) {
      setLocalColorOverride(settings.theme_color);
      return;
    }
    
    // Save settings to URL and cookies
    setUrlSettings(settings);
    setCookieSettings(settings);
    
    // Update local state
    setConfig((prev) => ({
      ...prev,
      settings: settings,
    }));
  }, [appConfig.settings.editable, setUrlSettings, setCookieSettings]);
  
  // Initialize config
  useEffect(() => {
    setConfig(getConfig());
  }, [getConfig]);
  
  // Create memoized context value
  const contextValue = useMemo(() => ({
    config,
    setUserSettings,
  }), [config, setUserSettings]);
  
  return (
    <ConfigContext.Provider value={contextValue}>
      {children}
    </ConfigContext.Provider>
  );
};

// Hook to use config
export const useConfig = () => {
  const context = useContext(ConfigContext);
  
  if (!context) {
    throw new Error("useConfig must be used within a ConfigProvider");
  }
  
  return context;
};
```

# src\hooks\useConfig.tsx

```tsx
// src/hooks/useConfig.tsx
"use client";

import { getCookie, setCookie } from "cookies-next";
import jsYaml from "js-yaml";
import { useRouter } from "next/navigation";
import React, {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

export type UserSettings = {
  editable: boolean;
  theme_color: string;
  chat: boolean;
  inputs: {
    camera: boolean;
    mic: boolean;
  };
  outputs: {
    audio: boolean;
    video: boolean;
  };
  ws_url: string;
  token: string;
  // Memory system settings
  memory_enabled: boolean;
  memory_ws_url: string;
  memory_hpc_url: string;
};

export type AppConfig = {
  title: string;
  description: string;
  github_link?: string;
  video_fit?: "cover" | "contain";
  settings: UserSettings;
  show_qr?: boolean;
};

// Fallback if NEXT_PUBLIC_APP_CONFIG is not set
const defaultConfig: AppConfig = {
  title: "SYNTHIENCE Neural Playground",
  description: "A lucid neural presence—real-time, memory-driven, and always aware.",
  video_fit: "cover",
  settings: {
    editable: true,
    theme_color: "cyan",
    chat: true,
    inputs: {
      camera: true,
      mic: true,
    },
    outputs: {
      audio: true,
      video: true,
    },
    ws_url: "",
    token: "",
    // Default memory settings
    memory_enabled: false,
    memory_ws_url: "ws://localhost:5001",
    memory_hpc_url: "ws://localhost:5005"
  },
  show_qr: false,
};

const useAppConfig = (): AppConfig => {
  return useMemo(() => {
    if (process.env.NEXT_PUBLIC_APP_CONFIG) {
      try {
        const parsedConfig = jsYaml.load(
          process.env.NEXT_PUBLIC_APP_CONFIG
        ) as AppConfig;
        // Default to missing values in parsed config
        if (parsedConfig.settings === undefined) {
          parsedConfig.settings = defaultConfig.settings;
        }
        if (parsedConfig.settings.editable === undefined) {
          parsedConfig.settings.editable = defaultConfig.settings.editable;
        }
        if (parsedConfig.settings.memory_enabled === undefined) {
          parsedConfig.settings.memory_enabled = defaultConfig.settings.memory_enabled;
        }
        if (parsedConfig.settings.memory_ws_url === undefined) {
          parsedConfig.settings.memory_ws_url = defaultConfig.settings.memory_ws_url;
        }
        if (parsedConfig.settings.memory_hpc_url === undefined) {
          parsedConfig.settings.memory_hpc_url = defaultConfig.settings.memory_hpc_url;
        }
        return parsedConfig;
      } catch (e) {
        console.error("Error parsing app config:", e);
      }
    }
    return defaultConfig;
  }, []);
};

type ConfigData = {
  config: AppConfig;
  setUserSettings: (settings: UserSettings) => void;
};

const ConfigContext = createContext<ConfigData | undefined>(undefined);

export const ConfigProvider = ({ children }: { children: React.ReactNode }) => {
  const appConfig = useAppConfig();
  const router = useRouter();
  const [localColorOverride, setLocalColorOverride] = useState<string | null>(
    null
  );

  const getSettingsFromUrl = useCallback(() => {
    if (typeof window === "undefined") {
      return null;
    }
    if (!window.location.hash) {
      return null;
    }
    const appConfigFromSettings = appConfig;
    if (appConfigFromSettings.settings.editable === false) {
      return null;
    }
    
    try {
      const params = new URLSearchParams(window.location.hash.replace("#", ""));
      
      // Create memory URL defaults if memory is enabled but URLs not specified
      const memoryEnabled = params.get("memory") === "1";
      const memoryWsUrl = params.get("memory_ws") || "ws://localhost:5001";
      const memoryHpcUrl = params.get("memory_hpc") || "ws://localhost:5005";
      
      console.log(`URL memory settings - enabled: ${memoryEnabled}, ws: ${memoryWsUrl}, hpc: ${memoryHpcUrl}`);
      
      return {
        editable: true,
        chat: params.get("chat") === "1",
        theme_color: params.get("theme_color") || "cyan", // Provide default value to avoid null
        inputs: {
          camera: params.get("cam") === "1",
          mic: params.get("mic") === "1",
        },
        outputs: {
          audio: params.get("audio") === "1",
          video: params.get("video") === "1",
        },
        ws_url: "",
        token: "",
        // Memory settings from URL
        memory_enabled: memoryEnabled,
        memory_ws_url: memoryWsUrl,
        memory_hpc_url: memoryHpcUrl
      };
    } catch (error) {
      console.error("Error parsing URL parameters:", error);
      return null;
    }
  }, [appConfig]);

  const getSettingsFromCookies = useCallback(() => {
    if (typeof window === 'undefined') {
      // Return default settings during server-side rendering
      return appConfig.settings;
    }
    
    const appConfigFromSettings = appConfig;
    if (appConfigFromSettings.settings.editable === false) {
      return null;
    }
    const jsonSettings = getCookie("lk_settings");
    console.log("Initial cookie settings:", jsonSettings);
    
    if (jsonSettings) {
      try {
        const parsedSettings = JSON.parse(jsonSettings as string);
        console.log("Parsed settings from cookies:", parsedSettings);
        return parsedSettings as UserSettings;
      } catch (e) {
        console.error('Error parsing stored settings:', e);
        return appConfigFromSettings.settings;
      }
    }
    return appConfigFromSettings.settings;
  }, [appConfig]);

  const setUrlSettings = useCallback(
    (us: UserSettings) => {
      if (typeof window === 'undefined') {
        // Skip URL updates during server-side rendering
        return;
      }
      
      try {
        const obj = new URLSearchParams({
          cam: boolToString(us.inputs.camera),
          mic: boolToString(us.inputs.mic),
          video: boolToString(us.outputs.video),
          audio: boolToString(us.outputs.audio),
          chat: boolToString(us.chat),
          theme_color: us.theme_color || "cyan",
          memory: boolToString(us.memory_enabled || false)
        });
        
        // Log the URL parameters for debugging
        console.log("URL settings:", obj.toString());
        
        // Add memory URLs if memory is enabled
        if (us.memory_enabled) {
          obj.set('memory_ws', us.memory_ws_url);
          obj.set('memory_hpc', us.memory_hpc_url);
        }
        
        // Note: We don't set ws_url and token to the URL on purpose
        if (typeof window !== 'undefined') {
          // Use window.location directly instead of router to avoid SSR issues
          const currentUrl = window.location.pathname + '#' + obj.toString();
          window.history.replaceState({}, '', currentUrl);
        }
      } catch (error) {
        console.error("Error updating URL settings:", error);
      }
    },
    [] // Remove router dependency since we're not using it anymore
  );

  const setCookieSettings = useCallback((us: UserSettings) => {
    if (typeof window === 'undefined') {
      // Skip cookie operations during server-side rendering
      return;
    }
    
    try {
      const json = JSON.stringify(us);
      setCookie("lk_settings", json);
      console.log("Saved settings to cookies:", json);
    } catch (error) {
      console.error("Error saving settings to cookies:", error);
    }
  }, []);

  const getConfig = useCallback(() => {
    const appConfigFromSettings = appConfig;

    if (appConfigFromSettings.settings.editable === false) {
      if (localColorOverride) {
        appConfigFromSettings.settings.theme_color = localColorOverride;
      }
      return appConfigFromSettings;
    }
    const cookieSettigs = getSettingsFromCookies();
    const urlSettings = getSettingsFromUrl();
    if (!cookieSettigs) {
      if (urlSettings) {
        setCookieSettings(urlSettings);
      }
    }
    if (!urlSettings) {
      if (cookieSettigs) {
        setUrlSettings(cookieSettigs);
      }
    }
    const newCookieSettings = getSettingsFromCookies();
    if (!newCookieSettings) {
      return appConfigFromSettings;
    }
    appConfigFromSettings.settings = newCookieSettings;
    return { ...appConfigFromSettings };
  }, [
    appConfig,
    getSettingsFromCookies,
    getSettingsFromUrl,
    localColorOverride,
    setCookieSettings,
    setUrlSettings,
  ]);

  const setUserSettings = useCallback(
    (settings: UserSettings) => {
      const appConfigFromSettings = appConfig;
      if (appConfigFromSettings.settings.editable === false) {
        setLocalColorOverride(settings.theme_color);
        return;
      }
      setUrlSettings(settings);
      setCookieSettings(settings);
      _setConfig((prev) => {
        return {
          ...prev,
          settings: settings,
        };
      });
    },
    [appConfig, setCookieSettings, setUrlSettings]
  );

  const [config, _setConfig] = useState<AppConfig>(getConfig());

  useEffect(() => {
    // Compare user settings in cookie with current state
    if (typeof window === 'undefined') {
      return; // Skip during server-side rendering
    }
    
    // Provide defaults for memory settings if they're undefined
    const currentSettings = config.settings;
    if (currentSettings.memory_enabled === undefined) {
      console.log('Initializing undefined memory_enabled to false');
      _setConfig(prev => ({
        ...prev,
        settings: {
          ...prev.settings,
          memory_enabled: false
        }
      }));
    }
    
    const storedSettings = getCookie('userSettings');
    
    if (storedSettings) {
      try {
        const parsedSettings = JSON.parse(storedSettings);
        
        // If there's a change from default for memory settings, update app config
        if (parsedSettings.memory_enabled !== undefined && 
            parsedSettings.memory_enabled !== currentSettings.memory_enabled) {
          console.log(`Updating memory_enabled from cookie: ${parsedSettings.memory_enabled}`);
          
          _setConfig(prev => ({
            ...prev,
            settings: {
              ...prev.settings,
              memory_enabled: parsedSettings.memory_enabled
            }
          }));
        }
      } catch (e) {
        console.error('Error parsing stored settings:', e);
      }
    }
  }, []);

  // Run things client side because we use cookies
  useEffect(() => {
    if (typeof window !== 'undefined') {
      _setConfig(getConfig());
    }
  }, [getConfig]);

  return (
    <ConfigContext.Provider value={{ config, setUserSettings }}>
      {children}
    </ConfigContext.Provider>
  );
};

export const useConfig = () => {
  const context = React.useContext(ConfigContext);
  if (context === undefined) {
    throw new Error("useConfig must be used within a ConfigProvider");
  }
  return context;
};

const boolToString = (b: boolean) => (b ? "1" : "0");
```

# src\hooks\useConnection.tsx

```tsx
"use client"

import { useCloud } from "@/cloud/useCloud";
import React, { createContext, useState } from "react";
import { useCallback } from "react";
import { useConfig } from "./useConfig";
import { useToast } from "@/components/toast/ToasterProvider";

export type ConnectionMode = "cloud" | "manual" | "env"

type TokenGeneratorData = {
  shouldConnect: boolean;
  wsUrl: string;
  token: string;
  mode: ConnectionMode;
  disconnect: () => Promise<void>;
  connect: (mode: ConnectionMode) => Promise<void>;
};

const ConnectionContext = createContext<TokenGeneratorData | undefined>(undefined);

export const ConnectionProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const { generateToken, wsUrl: cloudWSUrl } = useCloud();
  const { setToastMessage } = useToast();
  const { config } = useConfig();
  const [connectionDetails, setConnectionDetails] = useState<{
    wsUrl: string;
    token: string;
    mode: ConnectionMode;
    shouldConnect: boolean;
  }>({ wsUrl: "", token: "", shouldConnect: false, mode: "manual" });

  const connect = useCallback(
    async (mode: ConnectionMode) => {
      let token = "";
      let url = "";
      if (mode === "cloud") {
        try {
          token = await generateToken();
        } catch (error) {
          setToastMessage({
            type: "error",
            message:
              "Failed to generate token, you may need to increase your role in this LiveKit Cloud project.",
          });
        }
        url = cloudWSUrl;
      } else if (mode === "env") {
        if (!process.env.NEXT_PUBLIC_LIVEKIT_URL) {
          throw new Error("NEXT_PUBLIC_LIVEKIT_URL is not set");
        }
        url = process.env.NEXT_PUBLIC_LIVEKIT_URL;
        const { accessToken } = await fetch("/api/token").then((res) =>
          res.json()
        );
        token = accessToken;
      } else {
        token = config.settings.token;
        url = config.settings.ws_url;
      }
      setConnectionDetails({ wsUrl: url, token, shouldConnect: true, mode });
    },
    [
      cloudWSUrl,
      config.settings.token,
      config.settings.ws_url,
      generateToken,
      setToastMessage,
    ]
  );

  const disconnect = useCallback(async () => {
    setConnectionDetails((prev) => ({ ...prev, shouldConnect: false }));
  }, []);

  return (
    <ConnectionContext.Provider
      value={{
        wsUrl: connectionDetails.wsUrl,
        token: connectionDetails.token,
        shouldConnect: connectionDetails.shouldConnect,
        mode: connectionDetails.mode,
        connect,
        disconnect,
      }}
    >
      {children}
    </ConnectionContext.Provider>
  );
};

export const useConnection = () => {
  const context = React.useContext(ConnectionContext);
  if (context === undefined) {
    throw new Error("useConnection must be used within a ConnectionProvider");
  }
  return context;
}
```

# src\hooks\useMemory.tsx

```tsx
// src/hooks/useMemory.tsx
import { useState, useEffect, useCallback, useRef } from 'react';
import { getMemoryService } from '@/lib/memoryService';

// Define hook props with default URLs
interface UseMemoryProps {
  defaultTensorUrl?: string;
  defaultHpcUrl?: string;
  enabled?: boolean;
}

type MemoryMetric = {
  id: string;
  text: string;
  similarity: number;
  significance: number;
  surprise: number;
  timestamp: number;
};

type MemoryStats = {
  memory_count: number;
  gpu_memory: number;
  active_connections: number;
};

// Define memory hook
export const useMemory = ({ defaultTensorUrl, defaultHpcUrl, enabled }: UseMemoryProps = {}) => {
  console.log("useMemory hook initialized with URLs:", { defaultTensorUrl, defaultHpcUrl });
  
  // State for URLs
  const tensorUrlRef = useRef<string>(defaultTensorUrl || 'ws://localhost:5001');
  const hpcUrlRef = useRef<string>(defaultHpcUrl || 'ws://localhost:5005');

  // Memory enabled state
  const [memoryEnabled, setMemoryEnabled] = useState<boolean>(enabled !== undefined ? enabled : false);
  
  // Connection states
  const [connectionStatus, setConnectionStatus] = useState<string>('Disconnected');
  const [hpcStatus, setHpcStatus] = useState<string>('Disconnected');
  
  // Results and selection states
  const [memoryResults, setMemoryResults] = useState<MemoryMetric[]>([]);
  const [selectedMemories, setSelectedMemories] = useState<Set<string>>(new Set());
  
  // Stats and metrics
  const [stats, setStats] = useState<MemoryStats>({
    memory_count: 0,
    gpu_memory: 0,
    active_connections: 0
  });
  
  const [processingMetrics, setProcessingMetrics] = useState<{ 
    significance: number[], 
    surprise: number[] 
  }>({
    significance: Array(5).fill(0.5),
    surprise: Array(5).fill(0.25)
  });

  // Initialize memory service
  const initialize = useCallback((tensorUrl?: string, hpcUrl?: string) => {
    if (tensorUrl) tensorUrlRef.current = tensorUrl;
    if (hpcUrl) hpcUrlRef.current = hpcUrl;

    console.log(`Initializing memory service with enabled state: ${memoryEnabled}`);
    const memoryService = getMemoryService(tensorUrlRef.current, hpcUrlRef.current);
    
    // Make sure our local state is synced with the memory service
    setConnectionStatus('Connecting');
    memoryService.initialize()
      .then((success: boolean) => {
        if (success) {
          console.log('Memory system initialized successfully');
          
          // Set memory service enabled state based on our enabled state
          memoryService.setEnabled(memoryEnabled);
        } else {
          console.error('Failed to initialize memory system');
        }
      })
      .catch((error: Error) => {
        console.error('Error initializing memory system:', error);
        setConnectionStatus('Error');
      });
  }, [memoryEnabled]);

  // Disconnect memory service
  const disconnect = useCallback(() => {
    const memoryService = getMemoryService();
    memoryService.disconnect();
    setConnectionStatus('Disconnected');
    setHpcStatus('Disconnected');
  }, []);

  // Toggle connection
  const toggleConnection = useCallback(() => {
    if (connectionStatus === 'Connected') {
      disconnect();
    } else {
      initialize();
    }
  }, [connectionStatus, disconnect, initialize]);

  // Toggle memory system enabled/disabled
  const toggleMemorySystem = useCallback(() => {
    const memoryService = getMemoryService();
    
    const newEnabledState = !memoryEnabled;
    console.log(`Toggling memory system to: ${newEnabledState}`);
    
    memoryService.setEnabled(newEnabledState);
    // State will be updated through the event handler
  }, [memoryEnabled]);

  // Set search text (for external components)
  const setSearchText = useCallback((text: string) => {
    // Currently just a passthrough but could add preprocessing
    return text;
  }, []);

  // Search for memories
  const search = useCallback((query: string) => {
    if (!query.trim()) return;
    
    const memoryService = getMemoryService();
    memoryService.search(query);
  }, []);

  // Clear search results
  const clearSearch = useCallback(() => {
    setMemoryResults([]);
  }, []);

  // Toggle selection of a memory
  const toggleSelection = useCallback((id: string) => {
    setSelectedMemories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      
      // Update memory service with selection
      const memoryService = getMemoryService();
      memoryService.updateSelection(Array.from(newSet));
      
      return newSet;
    });
  }, []);

  // Set up event handlers
  useEffect(() => {
    const memoryService = getMemoryService(tensorUrlRef.current, hpcUrlRef.current);
    
    // Register event handlers
    const statusHandler = (data: any) => {
      setConnectionStatus(data.status);
    };
    
    const hpcStatusHandler = (data: any) => {
      setHpcStatus(data.status);
    };
    
    const searchResultsHandler = (data: any) => {
      if (data.results) {
        // Ensure each result has surprise metric (fallback to calculated value if not provided)
        const enhancedResults = data.results.map((result: any, index: number) => {
          // If surprise is not provided, generate one based on significance
          const surprise = result.surprise !== undefined ? 
            result.surprise : 
            Math.min(1.0, Math.max(0.1, result.significance * (1 + Math.random() * 0.5)));
            
          return {
            ...result,
            surprise
          };
        });
        
        setMemoryResults(enhancedResults);
      }
    };
    
    const statsHandler = (data: any) => {
      setStats(prev => ({
        ...prev,
        memory_count: data.memory_count || prev.memory_count,
        gpu_memory: data.gpu_memory || prev.gpu_memory,
        active_connections: data.active_connections || prev.active_connections
      }));
    };
    
    const selectionChangedHandler = (data: any) => {
      setSelectedMemories(new Set(data.selectedMemories));
    };
    
    const enabledChangedHandler = (data: any) => {
      setMemoryEnabled(data.enabled);
    };
    
    const memoryProcessedHandler = (data: any) => {
      if (data.significance !== undefined || data.surprise !== undefined) {
        setProcessingMetrics(prev => {
          // Add new metrics to the beginning of the arrays and keep only last 5
          const newSignificance = [...prev.significance];
          const newSurprise = [...prev.surprise];
          
          if (data.significance !== undefined) {
            newSignificance.unshift(data.significance);
            newSignificance.length = Math.min(newSignificance.length, 5);
          }
          
          if (data.surprise !== undefined) {
            newSurprise.unshift(data.surprise);
            newSurprise.length = Math.min(newSurprise.length, 5);
          }
          
          return {
            significance: newSignificance,
            surprise: newSurprise
          };
        });
      }
    };
    
    // Register callbacks
    memoryService.on('status', statusHandler);
    memoryService.on('hpc_status', hpcStatusHandler);
    memoryService.on('search_results', searchResultsHandler);
    memoryService.on('stats', statsHandler);
    memoryService.on('selection_changed', selectionChangedHandler);
    memoryService.on('enabled_changed', enabledChangedHandler);
    memoryService.on('memory_processed', memoryProcessedHandler);
    
    // Initialize if default URLs are provided
    if (defaultTensorUrl && defaultHpcUrl) {
      initialize(defaultTensorUrl, defaultHpcUrl);
    }
    
    // Cleanup on unmount
    return () => {
      memoryService.off('status', statusHandler);
      memoryService.off('hpc_status', hpcStatusHandler);
      memoryService.off('search_results', searchResultsHandler);
      memoryService.off('stats', statsHandler);
      memoryService.off('selection_changed', selectionChangedHandler);
      memoryService.off('enabled_changed', enabledChangedHandler);
      memoryService.off('memory_processed', memoryProcessedHandler);
    };
  }, [defaultTensorUrl, defaultHpcUrl, initialize]);

  return {
    connectionStatus,
    hpcStatus,
    memoryEnabled,
    setSearchText,
    search,
    clearSearch,
    selectedMemories,
    toggleSelection,
    stats,
    results: memoryResults,
    processingMetrics,
    toggleMemorySystem,
    
    // Return URLs
    memoryWsUrl: tensorUrlRef.current,
    memoryHpcUrl: hpcUrlRef.current,
    
    // For backward compatibility
    memoryResults,
    searchMemory: search,
    toggleMemorySelection: toggleSelection,
    clearSelectedMemories: clearSearch,
    processInput: (text: string) => console.log('Legacy processInput called', text),
    toggleConnection: () => console.log('Legacy toggleConnection called')
  };
}
```

# src\hooks\useTrackVolume.tsx

```tsx
import { Track } from "livekit-client";
import { useEffect, useState } from "react";

export const useTrackVolume = (track?: Track) => {
  const [volume, setVolume] = useState(0);
  useEffect(() => {
    if (!track || !track.mediaStream) {
      return;
    }

    const ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(track.mediaStream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 32;
    analyser.smoothingTimeConstant = 0;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const updateVolume = () => {
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const a = dataArray[i];
        sum += a * a;
      }
      setVolume(Math.sqrt(sum / dataArray.length) / 255);
    };

    const interval = setInterval(updateVolume, 1000 / 30);

    return () => {
      source.disconnect();
      clearInterval(interval);
    };
  }, [track, track?.mediaStream]);

  return volume;
};

const normalizeFrequencies = (frequencies: Float32Array) => {
  const normalizeDb = (value: number) => {
    const minDb = -100;
    const maxDb = -10;
    let db = 1 - (Math.max(minDb, Math.min(maxDb, value)) * -1) / 100;
    db = Math.sqrt(db);

    return db;
  };

  // Normalize all frequency values
  return frequencies.map((value) => {
    if (value === -Infinity) {
      return 0;
    }
    return normalizeDb(value);
  });
};

export const useMultibandTrackVolume = (
  track?: Track,
  bands: number = 5,
  loPass: number = 100,
  hiPass: number = 600
) => {
  const [frequencyBands, setFrequencyBands] = useState<Float32Array[]>([]);

  useEffect(() => {
    if (!track || !track.mediaStream) {
      return;
    }

    const ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(track.mediaStream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);

    const updateVolume = () => {
      analyser.getFloatFrequencyData(dataArray);
      let frequencies: Float32Array = new Float32Array(dataArray.length);
      for (let i = 0; i < dataArray.length; i++) {
        frequencies[i] = dataArray[i];
      }
      frequencies = frequencies.slice(loPass, hiPass);

      const normalizedFrequencies = normalizeFrequencies(frequencies);
      const chunkSize = Math.ceil(normalizedFrequencies.length / bands);
      const chunks: Float32Array[] = [];
      for (let i = 0; i < bands; i++) {
        chunks.push(
          normalizedFrequencies.slice(i * chunkSize, (i + 1) * chunkSize)
        );
      }

      setFrequencyBands(chunks);
    };

    const interval = setInterval(updateVolume, 10);

    return () => {
      source.disconnect();
      clearInterval(interval);
    };
  }, [track, track?.mediaStream, loPass, hiPass, bands]);

  return frequencyBands;
};

```

# src\hooks\useWindowResize.ts

```ts
import { useEffect, useState } from "react";

export const useWindowResize = () => {
  const [size, setSize] = useState({
    width: 0,
    height: 0,
  });

  useEffect(() => {
    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    handleResize();

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return size;
};

```

# src\lib\animations.ts

```ts
// src/lib/animations.ts
/**
 * Animation utilities for cyberpunk UI effects
 */

/**
 * Creates a neon glow trail effect that follows cursor movement on the element
 */
export const addGlowTrail = (element: HTMLElement) => {
  element.addEventListener('mousemove', (e) => {
    const rect = element.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const glow = document.createElement('div');
    glow.className = 'glow-trail';
    glow.style.position = 'absolute';
    glow.style.width = '8px';
    glow.style.height = '8px';
    glow.style.background = 'rgba(0, 255, 255, 0.5)';
    glow.style.borderRadius = '50%';
    glow.style.pointerEvents = 'none';
    glow.style.zIndex = '10';
    glow.style.left = `${x}px`;
    glow.style.top = `${y}px`;
    glow.style.transform = 'translate(-50%, -50%)';
    
    element.appendChild(glow);
    
    // Animate the glow trail
    setTimeout(() => {
      glow.style.transition = 'all 500ms ease-out';
      glow.style.opacity = '0';
      glow.style.width = '16px';
      glow.style.height = '16px';
    }, 10);
    
    // Remove the glow trail after animation
    setTimeout(() => {
      if (element.contains(glow)) {
        element.removeChild(glow);
      }
    }, 500);
  });
};

/**
 * Creates a distortion pulse effect when the element is clicked
 */
export const addDistortionPulse = (element: HTMLElement) => {
  element.addEventListener('click', () => {
    // Store original box-shadow
    const originalShadow = element.style.boxShadow;
    
    // Apply distortion effect
    element.style.transform = 'scale(0.95)';
    element.style.boxShadow = '0 0 15px rgba(0, 255, 255, 0.7)';
    
    // Add glitch effect
    const glitch = document.createElement('div');
    glitch.style.position = 'absolute';
    glitch.style.inset = '0';
    glitch.style.backgroundColor = 'rgba(0, 255, 255, 0.1)';
    glitch.style.zIndex = '1';
    glitch.style.pointerEvents = 'none';
    
    element.appendChild(glitch);
    
    // Reset after animation
    setTimeout(() => {
      element.style.transform = 'scale(1)';
      element.style.boxShadow = originalShadow;
      if (element.contains(glitch)) {
        element.removeChild(glitch);
      }
    }, 150);
  });
};

/**
 * Creates a cursor glow effect that follows the mouse
 */
export const createCursorGlow = () => {
  // Create cursor glow element
  const glow = document.createElement('div');
  glow.className = 'cursor-glow';
  glow.style.position = 'fixed';
  glow.style.width = '24px';
  glow.style.height = '24px';
  glow.style.borderRadius = '50%';
  glow.style.background = 'radial-gradient(circle, rgba(0,255,255,0.2) 0%, rgba(0,255,255,0) 70%)';
  glow.style.pointerEvents = 'none';
  glow.style.zIndex = '9999';
  glow.style.transform = 'translate(-50%, -50%)';
  
  document.body.appendChild(glow);
  
  // Update cursor glow position
  const updatePosition = (e: MouseEvent) => {
    glow.style.left = `${e.clientX}px`;
    glow.style.top = `${e.clientY}px`;
  };
  
  // Add event listener
  document.addEventListener('mousemove', updatePosition);
  
  // Return cleanup function
  return () => {
    document.removeEventListener('mousemove', updatePosition);
    if (document.body.contains(glow)) {
      document.body.removeChild(glow);
    }
  };
};

/**
 * Creates a flickering text effect
 */
export const createTextFlicker = (element: HTMLElement) => {
  const text = element.textContent || '';
  element.textContent = '';
  
  // Create wrapper for animation
  const wrapper = document.createElement('span');
  wrapper.style.position = 'relative';
  
  // Split text into individual spans for letter-by-letter animation
  [...text].forEach(char => {
    const span = document.createElement('span');
    span.textContent = char;
    span.style.display = 'inline-block';
    
    // Random flicker animation
    if (Math.random() > 0.7) {
      span.style.animation = `flicker ${Math.random() * 5 + 2}s infinite`;
    }
    
    wrapper.appendChild(span);
  });
  
  element.appendChild(wrapper);
};

/**
 * Generates a neural pathway particle effect for the element
 */
export const createNeuralParticles = (element: HTMLElement, count: number = 20) => {
  const container = document.createElement('div');
  container.className = 'particles-container';
  container.style.position = 'absolute';
  container.style.top = '0';
  container.style.left = '0';
  container.style.width = '100%';
  container.style.height = '100%';
  container.style.overflow = 'hidden';
  container.style.pointerEvents = 'none';
  container.style.zIndex = '0';
  
  element.appendChild(container);
  
  // Create particles
  for (let i = 0; i < count; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.position = 'absolute';
    
    // Random position
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * 100}%`;
    
    // Random size
    const size = Math.random() * 3 + 1;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    
    // Set style
    particle.style.background = 'rgba(0, 255, 255, 0.5)';
    particle.style.borderRadius = '50%';
    
    // Random animation delay
    particle.style.animationDelay = `${Math.random() * 10}s`;
    
    // Random animation duration
    particle.style.animationDuration = `${Math.random() * 8 + 7}s`;
    
    // Apply animation
    particle.style.animation = 'particleFlow 10s linear infinite';
    
    container.appendChild(particle);
  }
  
  // Return cleanup function
  return () => {
    if (element.contains(container)) {
      element.removeChild(container);
    }
  };
};

/**
 * Creates a scan line effect on the element
 */
export const addScanLineEffect = (element: HTMLElement) => {
  // Add scan line class if not already present
  if (!element.classList.contains('scan-line')) {
    element.classList.add('scan-line');
  }
  
  // Create scan line element if needed
  if (!element.querySelector('.scan-line-element')) {
    const scanLine = document.createElement('div');
    scanLine.className = 'scan-line-element';
    scanLine.style.position = 'absolute';
    scanLine.style.top = '0';
    scanLine.style.left = '0';
    scanLine.style.width = '100%';
    scanLine.style.height = '4px';
    scanLine.style.background = 'linear-gradient(90deg, transparent 0%, rgba(0, 255, 255, 0.2) 50%, transparent 100%)';
    scanLine.style.opacity = '0.5';
    scanLine.style.zIndex = '1';
    scanLine.style.pointerEvents = 'none';
    scanLine.style.animation = 'scanAnimation 3s linear infinite';
    
    element.appendChild(scanLine);
  }
  
  return () => {
    element.classList.remove('scan-line');
    const scanLine = element.querySelector('.scan-line-element');
    if (scanLine) {
      element.removeChild(scanLine);
    }
  };
};

/**
 * Creates a digital glitch effect
 */
export const createGlitchEffect = (element: HTMLElement, intensity: number = 1) => {
  const glitchInterval = setInterval(() => {
    if (Math.random() > 0.95) {
      const glitch = document.createElement('div');
      glitch.style.position = 'absolute';
      glitch.style.top = `${Math.random() * 100}%`;
      glitch.style.left = '0';
      glitch.style.right = '0';
      glitch.style.height = `${Math.random() * 5 + 1}px`;
      glitch.style.backgroundColor = 'rgba(0, 255, 255, 0.5)';
      glitch.style.zIndex = '5';
      glitch.style.transform = `translateX(${(Math.random() - 0.5) * 10}px)`;
      
      element.appendChild(glitch);
      
      setTimeout(() => {
        if (element.contains(glitch)) {
          element.removeChild(glitch);
        }
      }, 300 * intensity);
    }
  }, 2000 / intensity);
  
  return () => {
    clearInterval(glitchInterval);
  };
};
```

# src\lib\memoryService.ts

```ts
// src/lib/memoryService.ts

type MemoryMetric = {
    id: string;
    text: string;
    similarity: number;
    significance: number;
    surprise: number;
    timestamp: number;
  };
  
  type MemoryStats = {
    memory_count: number;
    gpu_memory: number;
    active_connections: number;
  };
  
  type MemoryCallback = (data: any) => void;
  
  class MemoryService {
    private tensorServer: WebSocket | null = null;
    private hpcServer: WebSocket | null = null;
    private status: string = 'Disconnected';
    private hpcStatus: string = 'Disconnected';
    private memoryCount: number = 0;
    private callbacks: Map<string, MemoryCallback[]> = new Map();
    private memoryCache: Map<string, any> = new Map();
    private selectedMemories: Set<string> = new Set();
    private reconnectInterval: NodeJS.Timeout | null = null;
    private retryCount: number = 0;
    private maxRetries: number = 5;
    private enabled: boolean = false;
  
    constructor(
      private tensorUrl: string = 'ws://localhost:5001',
      private hpcUrl: string = 'ws://localhost:5005'
    ) {}
  
    /**
     * Initialize the memory service connections
     */
    initialize(tensorUrl?: string, hpcUrl?: string): Promise<boolean> {
      // Update URLs if provided
      if (tensorUrl) this.tensorUrl = tensorUrl;
      if (hpcUrl) this.hpcUrl = hpcUrl;
      
      return new Promise((resolve) => {
        // Log the URLs being used
        console.log(`Initializing memory service with tensorUrl: ${this.tensorUrl}, hpcUrl: ${this.hpcUrl}`);
        
        this.connectTensorServer()
          .then(() => {
            this.connectHPCServer();
            resolve(true);
          })
          .catch(() => {
            resolve(false);
          });
      });
    }
  
    /**
     * Set whether the memory system is enabled
     */
    setEnabled(enabled: boolean): void {
      console.log(`Memory service enabled state changing from ${this.enabled} to ${enabled}`);
      if (this.enabled === enabled) {
        console.log('Memory service enabled state unchanged, skipping update');
        return;
      }
      
      this.enabled = enabled;
      this.emit('enabled_changed', { enabled });
      console.log(`Memory service enabled state set to: ${enabled}`);
      
      // When enabling, try to connect if not already connected
      if (enabled && this.status !== 'Connected') {
        console.log('Attempting to connect memory service after enabling');
        this.initialize();
      }
    }
  
    /**
     * Get whether the memory system is enabled
     */
    isEnabled(): boolean {
      return this.enabled;
    }
  
    /**
     * Update selection
     */
    updateSelection(selectedIds: string[]): void {
      this.selectedMemories = new Set(selectedIds);
      this.emit('selection_changed', { selectedMemories: Array.from(this.selectedMemories) });
    }
  
    /**
     * Search memory
     */
    search(query: string, limit: number = 5): boolean {
      console.log(`Searching for: "${query}" with limit ${limit}`);
      // Mock search results for now
      setTimeout(() => {
        if (!this.enabled) {
          console.log("Memory search skipped - memory system disabled");
          return;
        }
        
        const results = [
          { id: '1', text: 'Sample memory result 1 matching ' + query, significance: 0.85, surprise: 0.75 },
          { id: '2', text: 'Sample memory result 2 matching ' + query, significance: 0.65, surprise: 0.45 },
          { id: '3', text: 'Sample memory result 3 matching ' + query, significance: 0.55, surprise: 0.35 }
        ];
        
        this.emit('search_results', { results });
      }, 300);
      
      return true;
    }
  
    /**
     * Search memory (legacy alias)
     */
    searchMemory(query: string, limit: number = 5): boolean {
      return this.search(query, limit);
    }
  
    /**
     * Toggle memory selection
     */
    toggleMemorySelection(id: string): void {
      const newSelection = new Set(this.selectedMemories);
      if (newSelection.has(id)) {
        newSelection.delete(id);
      } else {
        newSelection.add(id);
      }
      this.updateSelection(Array.from(newSelection));
    }
  
    /**
     * Clear selected memories
     */
    clearSelectedMemories(): void {
      this.updateSelection([]);
    }
  
    /**
     * Connect to the tensor server
     */
    private connectTensorServer(): Promise<void> {
      return new Promise((resolve, reject) => {
        try {
          console.log(`Connecting to tensor server at ${this.tensorUrl}`);
          this.tensorServer = new WebSocket(this.tensorUrl);
  
          this.tensorServer.onopen = () => {
            console.log("Tensor Server Connected");
            this.status = 'Connected';
            this.retryCount = 0;
            this.emit('status', { status: this.status });
            
            // Request initial stats
            this.sendToTensorServer({
              type: 'get_stats'
            });
            
            resolve();
          };
  
          this.tensorServer.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              console.log("Tensor Server received:", data);
              
              if (data.type === 'embeddings') {
                this.processEmbedding(data.embeddings);
              } else if (data.type === 'search_results') {
                this.emit('search_results', data);
              } else if (data.type === 'stats') {
                this.updateMemoryStats(data);
              }
            } catch (error) {
              console.error('Error parsing tensor server message:', error);
            }
          };
  
          this.tensorServer.onclose = () => {
            console.log("Tensor Server Disconnected");
            this.status = 'Disconnected';
            this.emit('status', { status: this.status });
            this.tensorServer = null;
            
            this.attemptReconnect();
          };
  
          this.tensorServer.onerror = (error) => {
            console.error('Tensor Server error:', error);
            this.status = 'Error';
            this.emit('status', { status: this.status });
            reject(error);
          };
        } catch (error) {
          console.error('Failed to connect to tensor server:', error);
          this.status = 'Error';
          this.emit('status', { status: this.status });
          reject(error);
        }
      });
    }
  
    /**
     * Connect to the HPC server
     */
    private connectHPCServer(): void {
      try {
        console.log(`Connecting to HPC server at ${this.hpcUrl}`);
        this.hpcServer = new WebSocket(this.hpcUrl);
  
        this.hpcServer.onopen = () => {
          console.log("HPC Server Connected");
          this.hpcStatus = 'Connected';
          this.emit('hpc_status', { status: 'Connected' });
        };
  
        this.hpcServer.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("HPC Server received:", data);
            
            if (data.type === 'processed') {
              this.memoryCount++;
              this.emit('memory_count', { count: this.memoryCount });
  
              // If we received significance data, store it
              if (data.significance !== undefined) {
                this.emit('memory_processed', { 
                  significance: data.significance,
                  surprise: data.surprise || Math.random() * 0.5 // Fallback if not provided
                });
              }
            }
          } catch (error) {
            console.error('Error parsing HPC server message:', error);
          }
        };
  
        this.hpcServer.onclose = () => {
          console.log("HPC Server Disconnected");
          this.hpcStatus = 'Disconnected';
          this.emit('hpc_status', { status: 'Disconnected' });
          this.hpcServer = null;
        };
  
        this.hpcServer.onerror = (error) => {
          console.error('HPC Server error:', error);
          this.hpcStatus = 'Error';
          this.emit('hpc_status', { status: 'Error' });
        };
      } catch (error) {
        console.error('Failed to connect to HPC server:', error);
        this.hpcStatus = 'Error';
        this.emit('hpc_status', { status: 'Error' });
      }
    }
  
    /**
     * Process embedding data
     */
    private async processEmbedding(embeddings: number[]): Promise<void> {
      try {
        // Send to HPC server
        if (this.hpcServer?.readyState === WebSocket.OPEN) {
          const hpcRequest = {
            type: 'process',
            embeddings: embeddings
          };
          this.hpcServer.send(JSON.stringify(hpcRequest));
        }
        
        // Cache locally with timestamp
        const timestamp = Date.now();
        this.memoryCache.set(timestamp.toString(), embeddings);
        
        // Update status
        this.status = 'Processing';
        this.emit('status', { status: this.status });
      } catch (error) {
        console.error('Error processing embedding:', error);
      }
    }
  
    /**
     * Update memory statistics
     */
    private updateMemoryStats(stats: MemoryStats): void {
      if (stats.memory_count !== undefined) {
        this.memoryCount = stats.memory_count;
      }
      
      this.emit('stats', stats);
    }
  
    /**
     * Send data to tensor server
     */
    sendToTensorServer(data: any): boolean {
      if (this.tensorServer?.readyState === WebSocket.OPEN) {
        this.tensorServer.send(JSON.stringify(data));
        return true;
      }
      return false;
    }
  
    /**
     * Process user input
     */
    processInput(text: string): Promise<boolean> {
      return new Promise((resolve) => {
        if (!text.trim() || !this.enabled) {
          resolve(false);
          return;
        }
  
        // Store in memory (on Tensor server)
        const storeSuccess = this.sendToTensorServer({
          type: 'embed',
          text: text
        });
  
        // Search memory
        const searchSuccess = this.sendToTensorServer({
          type: 'search',
          text: text,
          limit: 5
        });
  
        resolve(storeSuccess && searchSuccess);
      });
    }
  
    /**
     * Get selected memories
     */
    getSelectedMemories(): string[] {
      return Array.from(this.selectedMemories);
    }
  
    /**
     * Register callback for events
     */
    on(event: string, callback: MemoryCallback): void {
      if (!this.callbacks.has(event)) {
        this.callbacks.set(event, []);
      }
      
      this.callbacks.get(event)?.push(callback);
    }
  
    /**
     * Unregister callback
     */
    off(event: string, callback: MemoryCallback): void {
      const callbacks = this.callbacks.get(event);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index !== -1) {
          callbacks.splice(index, 1);
        }
      }
    }
  
    /**
     * Emit event
     */
    private emit(event: string, data: any): void {
      const callbacks = this.callbacks.get(event);
      if (callbacks) {
        callbacks.forEach(callback => callback(data));
      }
    }
  
    /**
     * Attempt to reconnect to servers
     */
    private attemptReconnect(): void {
      if (this.reconnectInterval) {
        clearInterval(this.reconnectInterval);
      }
      
      this.reconnectInterval = setInterval(() => {
        this.retryCount++;
        console.log(`Attempting to reconnect... (${this.retryCount}/${this.maxRetries})`);
        
        if (this.retryCount > this.maxRetries) {
          if (this.reconnectInterval) {
            clearInterval(this.reconnectInterval);
            this.reconnectInterval = null;
          }
          console.log('Max retry attempts reached');
          return;
        }
        
        if (!this.tensorServer) {
          this.connectTensorServer()
            .then(() => {
              if (this.reconnectInterval) {
                clearInterval(this.reconnectInterval);
                this.reconnectInterval = null;
              }
            })
            .catch(() => {});
        }
        
        if (!this.hpcServer) {
          this.connectHPCServer();
        }
      }, 5000);
    }
  
    /**
     * Disconnect from servers
     */
    disconnect(): void {
      if (this.tensorServer) {
        this.tensorServer.close();
        this.tensorServer = null;
      }
      
      if (this.hpcServer) {
        this.hpcServer.close();
        this.hpcServer = null;
      }
      
      if (this.reconnectInterval) {
        clearInterval(this.reconnectInterval);
        this.reconnectInterval = null;
      }
      
      this.status = 'Disconnected';
      this.hpcStatus = 'Disconnected';
      this.emit('status', { status: this.status });
      this.emit('hpc_status', { status: this.hpcStatus });
    }
  
    /**
     * Get connection status
     */
    getStatus(): string {
      return this.status;
    }
  
    /**
     * Get HPC status
     */
    getHPCStatus(): string {
      return this.hpcStatus;
    }
  }
  
  // Singleton instance
  let memoryServiceInstance: MemoryService | null = null;
  
  /**
   * Get memory service instance
   */
  export const getMemoryService = (
    tensorUrl?: string,
    hpcUrl?: string
  ): MemoryService => {
    if (!memoryServiceInstance) {
      memoryServiceInstance = new MemoryService(tensorUrl, hpcUrl);
    } else if (tensorUrl || hpcUrl) {
      // Update the URLs in the existing instance if provided
      memoryServiceInstance.initialize(tensorUrl, hpcUrl);
    }
    return memoryServiceInstance;
  };
```

# src\lib\tailwindTheme.preval.ts

```ts
import preval from "next-plugin-preval";
import resolveConfig from "tailwindcss/resolveConfig";
import tailwindConfig from "../../tailwind.config.js";

async function getTheme() {
  const fullTWConfig = resolveConfig(tailwindConfig);
  return fullTWConfig.theme;
}

export default preval(getTheme());

```

# src\lib\types.ts

```ts
import { LocalAudioTrack, LocalVideoTrack } from "livekit-client";

export interface SessionProps {
  roomName: string;
  identity: string;
  audioTrack?: LocalAudioTrack;
  videoTrack?: LocalVideoTrack;
  region?: string;
  turnServer?: RTCIceServer;
  forceRelay?: boolean;
}

export interface TokenResult {
  identity: string;
  accessToken: string;
}
```

# src\lib\util.ts

```ts
export function generateRandomAlphanumeric(length: number): string {
  const characters =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  const charactersLength = characters.length;

  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }

  return result;
}

```

# src\pages\_app.tsx

```tsx
import { CloudProvider } from "@/cloud/useCloud";
import "@livekit/components-styles/components/participant";
import "@/styles/globals.css";
import type { AppProps } from "next/app";

export default function App({ Component, pageProps }: AppProps) {
  return (
    <CloudProvider>
      <Component {...pageProps} />
    </CloudProvider>
  );
}

```

# src\pages\_document.tsx

```tsx
import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="en">
      <Head />
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}

```

# src\pages\api\token.ts

```ts
import { NextApiRequest, NextApiResponse } from "next";
import { generateRandomAlphanumeric } from "@/lib/util";

import { AccessToken } from "livekit-server-sdk";
import type { AccessTokenOptions, VideoGrant } from "livekit-server-sdk";
import { TokenResult } from "../../lib/types";

const apiKey = process.env.LIVEKIT_API_KEY;
const apiSecret = process.env.LIVEKIT_API_SECRET;

const createToken = (userInfo: AccessTokenOptions, grant: VideoGrant) => {
  const at = new AccessToken(apiKey, apiSecret, userInfo);
  at.addGrant(grant);
  return at.toJwt();
};

export default async function handleToken(
  req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    if (!apiKey || !apiSecret) {
      res.statusMessage = "Environment variables aren't set up correctly";
      res.status(500).end();
      return;
    }

    const roomName = `room-${generateRandomAlphanumeric(4)}-${generateRandomAlphanumeric(4)}`;
    const identity = `identity-${generateRandomAlphanumeric(4)}`

    const grant: VideoGrant = {
      room: roomName,
      roomJoin: true,
      canPublish: true,
      canPublishData: true,
      canSubscribe: true,
    };

    const token = await createToken({ identity }, grant);
    const result: TokenResult = {
      identity,
      accessToken: token,
    };

    res.status(200).json(result);
  } catch (e) {
    res.statusMessage = (e as Error).message;
    res.status(500).end();
  }
}
```

# src\pages\index.tsx

```tsx
import { LiveKitRoom, RoomAudioRenderer, StartAudio } from "@livekit/components-react";
import { AnimatePresence, motion } from "framer-motion";
import { Inter } from "next/font/google";
import Head from "next/head";
import { useCallback, useEffect, useRef } from "react";

import { PlaygroundConnect } from "@/components/PlaygroundConnect";
import Playground from "@/components/playground/Playground";
import { PlaygroundToast } from "@/components/toast/PlaygroundToast";
import { ConfigProvider, useConfig } from "@/hooks/useConfig";
import { ConnectionMode, ConnectionProvider, useConnection } from "@/hooks/useConnection";
import { useMemo } from "react";
import { ToastProvider, useToast } from "@/components/toast/ToasterProvider";
import { createNeuralParticles, createGlitchEffect } from "@/lib/animations";
import { NeuralInterfaceAnimation } from '@/components/cyberpunk/NeuralInterfaceAnimation';

const themeColors = [
  "cyan",
  "green",
  "amber",
  "blue",
  "violet",
  "rose",
  "pink",
  "teal",
];

const inter = Inter({ subsets: ["latin"] });

export default function Home() {
  return (
    <ToastProvider>
      <ConfigProvider>
        <ConnectionProvider>
          <HomeInner />
        </ConnectionProvider>
      </ConfigProvider>
    </ToastProvider>
  );
}

export function HomeInner() {
  const { shouldConnect, wsUrl, token, mode, connect, disconnect } = useConnection();
  const { config } = useConfig();
  const { toastMessage, setToastMessage } = useToast();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Apply cyberpunk effects to the main container
    if (containerRef.current) {
      // Neural particles effect
      const cleanupParticles = createNeuralParticles(containerRef.current, 20);
      
      // Occasional glitch effect
      const cleanupGlitch = createGlitchEffect(containerRef.current, 0.5);
      
      return () => {
        cleanupParticles();
        cleanupGlitch();
      };
    }
  }, []);

  const handleConnect = useCallback(
    async (c: boolean, mode: ConnectionMode) => {
      c ? connect(mode) : disconnect();
    },
    [connect, disconnect]
  );

  const showPG = useMemo(() => {
    if (process.env.NEXT_PUBLIC_LIVEKIT_URL) {
      return true;
    }
    if (wsUrl) {
      return true;
    }
    return false;
  }, [wsUrl]);

  return (
    <>
      <Head>
        <title>Synthience.AI | Neural Interface</title>
        <meta name="description" content={config.description || "Next-generation neural interface powered by advanced AI"} />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"
        />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black" />
        <meta
          property="og:image"
          content="https://livekit.io/images/og/agents-playground.png"
        />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <main 
        ref={containerRef}
        className="relative flex flex-col justify-center px-4 items-center h-full w-full bg-black repeating-square-background overflow-hidden"
      >
        <NeuralInterfaceAnimation />
        {/* Global holographic grid overlay */}
        <div className="absolute inset-0 pointer-events-none holo-grid"></div>
        
        <AnimatePresence>
          {toastMessage && (
            <motion.div
              className="left-0 right-0 top-0 absolute z-10"
              initial={{ opacity: 0, translateY: -50 }}
              animate={{ opacity: 1, translateY: 0 }}
              exit={{ opacity: 0, translateY: -50 }}
            >
              <PlaygroundToast />
            </motion.div>
          )}
        </AnimatePresence>
        
        {showPG ? (
          <LiveKitRoom
            className="flex flex-col h-full w-full"
            serverUrl={wsUrl}
            token={token}
            connect={shouldConnect}
            onError={(e) => {
              setToastMessage({ message: `Neural link error: ${e.message}`, type: "error" });
              console.error(e);
            }}
          >
            <Playground
              themeColors={themeColors}
              onConnect={(c) => {
                const m = process.env.NEXT_PUBLIC_LIVEKIT_URL ? "env" : mode;
                handleConnect(c, m);
              }}
            />
            <RoomAudioRenderer />
            <StartAudio label="Initialize audio processing" />
          </LiveKitRoom>
        ) : (
          <PlaygroundConnect
            accentColor={themeColors[0]}
            onConnectClicked={(mode) => {
              handleConnect(true, mode);
            }}
          />
        )}
      </main>
    </>
  );
}
```

# src\styles\globals.css

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --neon-cyan: rgba(0, 255, 255, 0.8);
  --neon-cyan-dim: rgba(0, 255, 255, 0.3);
  --neon-blue: rgba(0, 94, 255, 0.8);
  --neon-amber: rgba(255, 191, 0, 0.8);
  --neon-green: rgba(0, 255, 127, 0.8);
  --glass-bg: rgba(0, 0, 0, 0.3);
  --glass-border: rgba(0, 255, 255, 0.3);
}

body {
  background: #000;
  color: #e0e0e0;
  font-family: 'JetBrains Mono', monospace;
  --lk-va-bar-gap: 4px;
  --lk-va-bar-width: 4px;
  --lk-va-border-radius: 2px;
}

#__next {
  width: 100%;
  height: 100dvh;
  position: relative;
}

/* Cyberpunk Grid Background */
.repeating-square-background {
  background-size: 35px 35px;
  background-repeat: repeat;
  background-image: 
    linear-gradient(to right, rgba(0, 255, 255, 0.03) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
  position: relative;
  overflow: hidden;
}

.repeating-square-background::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at 50% 50%, transparent 0%, rgba(0, 0, 0, 0.8) 70%);
  pointer-events: none;
}

/* Holographic Grid Effect */
.holo-grid {
  position: relative;
}

.holo-grid::before {
  content: '';
  position: absolute;
  inset: 0;
  background-image: 
    linear-gradient(to right, transparent 95%, var(--neon-cyan-dim) 100%),
    linear-gradient(to bottom, transparent 95%, var(--neon-cyan-dim) 100%);
  background-size: 50px 50px;
  background-position: center;
  opacity: 0.15;
  z-index: -1;
  animation: gridFlow 15s linear infinite;
}

@keyframes gridFlow {
  0% { background-position: 0% 0%; }
  100% { background-position: 100% 100%; }
}

/* Glassmorphism Panel */
.glass-panel {
  background: var(--glass-bg);
  backdrop-filter: blur(8px);
  border: 1px solid var(--glass-border);
  border-radius: 4px;
  box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}

.glass-panel:hover {
  box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
}

/* Neon Glow Effects */
.text-glow {
  text-shadow: 0 0 5px var(--neon-cyan), 0 0 10px var(--neon-cyan);
}

.border-glow {
  box-shadow: 0 0 5px var(--neon-cyan), 0 0 10px var(--neon-cyan);
}

.border-glow-hover:hover {
  box-shadow: 0 0 10px var(--neon-cyan), 0 0 20px var(--neon-cyan);
}

/* Scan Line Effect */
.scan-line {
  position: relative;
  overflow: hidden;
}

.scan-line::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(
    90deg, 
    transparent 0%, 
    rgba(0, 255, 255, 0.2) 50%,
    transparent 100%
  );
  opacity: 0.5;
  animation: scanAnimation 3s linear infinite;
  z-index: 1;
  pointer-events: none;
}

@keyframes scanAnimation {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(2000%); }
}

/* Digital Flicker */
.digital-flicker {
  animation: flicker 4s linear infinite;
}

@keyframes flicker {
  0% { opacity: 1; }
  1% { opacity: 0.8; }
  2% { opacity: 1; }
  67% { opacity: 1; }
  68% { opacity: 0.8; }
  69% { opacity: 1; }
  70% { opacity: 1; }
  71% { opacity: 0.8; }
  72% { opacity: 1; }
}

/* Neural Pathway Particles */
.particles-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  pointer-events: none;
  z-index: 0;
}

.particle {
  position: absolute;
  width: 2px;
  height: 2px;
  background: var(--neon-cyan);
  border-radius: 50%;
  animation: particleFlow 10s linear infinite;
  opacity: 0.4;
}

@keyframes particleFlow {
  0% { transform: translate(0, 0); opacity: 0; }
  20% { opacity: 0.7; }
  80% { opacity: 0.7; }
  100% { transform: translate(100%, 100%); opacity: 0; }
}

/* Cursor Animation */
.cursor-animation {
  animation: cursor-blink 0.8s ease-in-out infinite alternate;
}

@keyframes cursor-blink {
  0% { opacity: 1; }
  100% { opacity: 0.3; }
}

/* Dashboard HUD Effects */
.hud-panel {
  position: relative;
  border: 1px solid var(--neon-cyan-dim);
  border-radius: 4px;
  background: rgba(0, 20, 40, 0.3);
  padding: 1rem;
  box-shadow: inset 0 0 15px rgba(0, 255, 255, 0.1);
}

.hud-panel::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: 
    linear-gradient(90deg, transparent 50%, rgba(0, 255, 255, 0.05) 51%, transparent 52%) 0 0 / 100px 100%,
    linear-gradient(0deg, transparent 50%, rgba(0, 255, 255, 0.05) 51%, transparent 52%) 0 0 / 100% 100px;
  pointer-events: none;
}

/* Loading Effect */
.cyber-loading {
  position: relative;
}

.cyber-loading::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, var(--neon-cyan-dim), transparent);
  animation: loading-sweep 1.5s infinite;
}

@keyframes loading-sweep {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

/* Scrollbar Styling */
::-webkit-scrollbar-track {
  background: rgba(0, 20, 40, 0.3);
  border-radius: 5px;
}

::-webkit-scrollbar-thumb {
  background: rgba(0, 255, 255, 0.2);
  border-radius: 5px;
  border: 1px solid rgba(0, 255, 255, 0.1);
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 255, 255, 0.4);
}

::-webkit-scrollbar {
  width: 8px;
  border-radius: 5px;
}

/* Status Indicators */
.status-active {
  position: relative;
}

.status-active::before {
  content: '';
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--neon-green);
  margin-right: 6px;
  box-shadow: 0 0 8px var(--neon-green);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 0.7; }
  50% { opacity: 1; }
  100% { opacity: 0.7; }
}

/* Add these styles to your globals.css file */

/* Memory System Specific Styles */
.memory-result {
  position: relative;
  padding: 1rem;
  background: rgba(10, 10, 31, 0.7);
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 4px;
  margin-bottom: 0.75rem;
  transition: all 0.3s ease;
}

.memory-result:hover {
  border-color: rgba(0, 255, 255, 0.5);
  transform: translateX(2px);
}

.memory-result.selected {
  background: rgba(255, 0, 255, 0.05);
  border-color: rgba(255, 0, 255, 0.4);
}

.memory-metric {
  display: inline-block;
  padding: 0.2rem 0.5rem;
  font-size: 0.75rem;
  font-family: 'JetBrains Mono', monospace;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 3px;
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
}

.metric-similarity {
  color: #00ffff;
  background: rgba(0, 255, 255, 0.1);
}

.metric-significance {
  color: #ffa500;
  background: rgba(255, 165, 0, 0.1);
}

.metric-surprise {
  color: #ff00ff;
  background: rgba(255, 0, 255, 0.1);
}

/* Particle Animation */
@keyframes particleFlow {
  0% { 
    transform: translate(0, 0); 
    opacity: 0; 
  }
  20% { 
    opacity: 0.7; 
  }
  80% { 
    opacity: 0.7; 
  }
  100% { 
    transform: translate(100px, 100px); 
    opacity: 0; 
  }
}

.particle {
  position: absolute;
  background: rgba(0, 255, 255, 0.5);
  border-radius: 50%;
  animation: particleFlow 10s linear infinite;
}

/* Neural Activity Visualization */
.neural-activity-container {
  position: relative;
  width: 100%;
  height: 150px;
  border: 1px solid rgba(0, 255, 255, 0.2);
  border-radius: 4px;
  background: rgba(0, 0, 0, 0.3);
  overflow: hidden;
}

.neural-node {
  position: absolute;
  width: 4px;
  height: 4px;
  background: rgba(0, 255, 255, 0.7);
  border-radius: 50%;
  transform: translate(-50%, -50%);
}

.neural-connection {
  position: absolute;
  height: 1px;
  background: rgba(0, 255, 255, 0.3);
  transform-origin: 0 0;
  z-index: 0;
}

/* Pulse Animation */
@keyframes pulse {
  0% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.1); opacity: 0.7; }
  100% { transform: scale(1); opacity: 1; }
}

.pulse {
  animation: pulse 2s ease-in-out infinite;
}

/* Memory Metrics Chart */
.bar-chart {
  display: flex;
  height: 100px;
  align-items: flex-end;
  justify-content: space-around;
  padding: 1rem;
  background: rgba(10, 10, 31, 0.5);
  border-radius: 4px;
  border: 1px solid rgba(0, 255, 255, 0.2);
}

.bar {
  width: 20px;
  background: linear-gradient(to top, rgba(0, 255, 255, 0.8), rgba(0, 255, 255, 0.2));
  border-radius: 2px 2px 0 0;
  transition: height 0.5s ease;
}

.bar-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.bar-label {
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: rgba(0, 255, 255, 0.8);
}

/* Add to the existing scan-line effect */
.scan-line::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: linear-gradient(
    90deg, 
    transparent 0%, 
    rgba(0, 255, 255, 0.2) 50%,
    transparent 100%
  );
  opacity: 0.5;
  animation: scanAnimation 3s linear infinite;
  z-index: 1;
  pointer-events: none;
}

@keyframes scanAnimation {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(2000%); }
}

/* Digital Flicker Animation */
@keyframes flicker {
  0% { opacity: 1; }
  1% { opacity: 0.8; }
  2% { opacity: 1; }
  67% { opacity: 1; }
  68% { opacity: 0.8; }
  69% { opacity: 1; }
  70% { opacity: 1; }
  71% { opacity: 0.8; }
  72% { opacity: 1; }
}

.digital-flicker {
  animation: flicker 4s linear infinite;
}
```

# src\transcriptions\TranscriptionTile.tsx

```tsx
import { ChatMessageType, ChatTile } from "@/components/chat/ChatTile";
import {
  TrackReferenceOrPlaceholder,
  useChat,
  useLocalParticipant,
  useTrackTranscription,
} from "@livekit/components-react";
import {
  LocalParticipant,
  Participant,
  Track,
  TranscriptionSegment,
} from "livekit-client";
import { useEffect, useState } from "react";

export function TranscriptionTile({
  agentAudioTrack,
  accentColor,
}: {
  agentAudioTrack: TrackReferenceOrPlaceholder;
  accentColor: string;
}) {
  const agentMessages = useTrackTranscription(agentAudioTrack);
  const localParticipant = useLocalParticipant();
  const localMessages = useTrackTranscription({
    publication: localParticipant.microphoneTrack,
    source: Track.Source.Microphone,
    participant: localParticipant.localParticipant,
  });

  const [transcripts, setTranscripts] = useState<Map<string, ChatMessageType>>(
    new Map()
  );
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const { chatMessages, send: sendChat } = useChat();

  // store transcripts
  useEffect(() => {
    agentMessages.segments.forEach((s) =>
      transcripts.set(
        s.id,
        segmentToChatMessage(
          s,
          transcripts.get(s.id),
          agentAudioTrack.participant
        )
      )
    );
    localMessages.segments.forEach((s) =>
      transcripts.set(
        s.id,
        segmentToChatMessage(
          s,
          transcripts.get(s.id),
          localParticipant.localParticipant
        )
      )
    );

    const allMessages = Array.from(transcripts.values());
    for (const msg of chatMessages) {
      const isAgent =
        msg.from?.identity === agentAudioTrack.participant?.identity;
      const isSelf =
        msg.from?.identity === localParticipant.localParticipant.identity;
      let name = msg.from?.name;
      if (!name) {
        if (isAgent) {
          name = "Agent";
        } else if (isSelf) {
          name = "You";
        } else {
          name = "Unknown";
        }
      }
      allMessages.push({
        name,
        message: msg.message,
        timestamp: msg.timestamp,
        isSelf: isSelf,
      });
    }
    allMessages.sort((a, b) => a.timestamp - b.timestamp);
    setMessages(allMessages);
  }, [
    transcripts,
    chatMessages,
    localParticipant.localParticipant,
    agentAudioTrack.participant,
    agentMessages.segments,
    localMessages.segments,
  ]);

  return (
    <ChatTile messages={messages} accentColor={accentColor} onSend={sendChat} />
  );
}

function segmentToChatMessage(
  s: TranscriptionSegment,
  existingMessage: ChatMessageType | undefined,
  participant: Participant
): ChatMessageType {
  console.log("Processing transcription segment:", {
    segment: s,
    participant: {
      identity: participant?.identity,
      name: participant?.name,
      isLocal: participant instanceof LocalParticipant
    }
  });
  
  // Determine if this is a user message based on participant type
  const isLocalParticipant = participant instanceof LocalParticipant;
  
  // Set name and isSelf based on participant identity
  let name = "Unknown";
  let isSelf = false;
  
  if (isLocalParticipant) {
    name = "You";
    isSelf = true;
  } else {
    name = "Agent";
    isSelf = false;
  }
  
  console.log(`Identified segment as ${isSelf ? "user" : "agent"} message:`, s.text);
  
  const msg: ChatMessageType = {
    message: s.final ? s.text : `${s.text} ...`,
    name: name,
    isSelf: isSelf,
    timestamp: existingMessage?.timestamp ?? Date.now(),
  };
  return msg;
}

```

# tailwind.config.js

```js
/** @type {import('tailwindcss').Config} */

const colors = require('tailwindcss/colors')
const shades = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900', '950'];
const colorList = ['gray', 'green', 'cyan', 'amber', 'violet', 'blue', 'rose', 'pink', 'teal', "red"];
const uiElements = ['bg', 'selection:bg', 'border', 'text', 'hover:bg', 'hover:border', 'hover:text', 'ring', 'focus:ring'];
const customColors = {
  cyan: colors.cyan,
  green: colors.green,
  amber: colors.amber,
  violet: colors.violet,
  blue: colors.blue,
  rose: colors.rose,
  pink: colors.pink,
  teal: colors.teal,
  red: colors.red,
};

let customShadows = {};
let shadowNames = [];
let textShadows = {};
let textShadowNames = [];

for (const [name, color] of Object.entries(customColors)) {
  customShadows[`${name}`] = `0px 0px 10px ${color["500"]}`;
  customShadows[`lg-${name}`] = `0px 0px 20px ${color["600"]}`;
  textShadows[`${name}`] = `0px 0px 4px ${color["700"]}`;
  textShadowNames.push(`drop-shadow-${name}`);
  shadowNames.push(`shadow-${name}`);
  shadowNames.push(`shadow-lg-${name}`);
  shadowNames.push(`hover:shadow-${name}`);
}

const safelist = [
  'bg-black',
  'bg-white',
  'transparent',
  'object-cover',
  'object-contain',
  ...shadowNames,
  ...textShadowNames,
  ...shades.flatMap(shade => [
    ...colorList.flatMap(color => [
      ...uiElements.flatMap(element => [
        `${element}-${color}-${shade}`,
      ]),
    ]),
  ]),
];

module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      black: colors.black,
      white: colors.white,
      gray: colors.neutral,
      ...customColors
    },
    extend: {
      dropShadow: {
       ...textShadows,
      },
      boxShadow: {
        ...customShadows,
      }
    }
  },
  plugins: [],
  safelist,
};
```

# tsconfig.json

```json
{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "downlevelIteration": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}

```

