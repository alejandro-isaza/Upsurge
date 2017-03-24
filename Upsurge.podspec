Pod::Spec.new do |s|
  s.name = 'Upsurge'
  s.version = '0.8.0'
  s.license = 'MIT'
  s.summary = 'Swift + Accelerate'
  s.homepage = 'https://github.com/aleph7/Upsurge'
  s.social_media_url = 'http://twitter.com/aleph7'
  s.authors = { 'Alejandro Isaza' => 'al@isaza.ca'  }
  s.source = { git: 'https://github.com/aleph7/Upsurge.git', tag: s.version }

  s.ios.deployment_target = '9.0'
  s.osx.deployment_target = '10.10'
  s.tvos.deployment_target = '9.0'

  s.source_files = 'Source/*.swift'

  s.frameworks = 'Accelerate'
end
